# Some constructor may requires calibration data to be passed in, so we need to
# write a forward hook to automatically collect the data.
import torch 
import torch.nn as nn 

# handle output in tuple, if tuple, only get the first result
def get_first_output(output):
    if type(output) == tuple:
        return output[0]
    else:
        return output



class CalibHelper:
    def __init__(self, model) -> None:
        self.forward_hook_targets = [nn.Linear, nn.Conv2d]
        self.assign_unique_id_to_all_layers(model)
        pass
    

    def assign_unique_id_to_all_layers(self, model:nn.Module):
        id_to_layer = {}
        # assign a unique id to each layer
        def assin_inner(module, prefix=''):
            for name, child in module.named_children():
                if not hasattr(child, 'unique_id'):
                    child.unique_id = prefix + name
                    id_to_layer[child.unique_id] = child
                    assin_inner(child, prefix + name + '.') # recursive call
                else:
                    continue
        # assign names to all models
        assin_inner(model)
        self.id_to_layer = id_to_layer
    
    def torch_int_forward_hook(self, module, input, output):
        unique_id = module.unique_id
        input_record = get_first_output(input)
        # running mean
        if unique_id not in self.collected_calib_data:
            self.collected_calib_data[unique_id] = input_record.detach().cpu() # store in cpu
        else:
            self.collected_calib_data[unique_id] = 0.9 * self.collected_calib_data[unique_id] + 0.1 * input_record.detach().cpu()
        return output
    
    def register_forward_hooks(self):
        self.fwd_hooks = []
        for layer_id, layer in self.id_to_layer.items():
            if type(layer) in self.forward_hook_targets:
                hook = layer.register_forward_hook(self.torch_int_forward_hook)
                self.fwd_hooks.append(hook)
        self.collected_calib_data = {}

    def remove_forward_hooks(self):
        for hook in self.fwd_hooks:
            hook.remove()

    
    def get_module_calib_data(self, module):
        return self.collected_calib_data[module.unique_id]

    # for some case, we may need to let users 
    def set_module_calib_data_to_module(self):
        for unique_id, calib_data in self.collected_calib_data.items():
            self.id_to_layer[unique_id].calib_data = calib_data
            self.id_to_layer[unique_id].has_calib_data = True

    
