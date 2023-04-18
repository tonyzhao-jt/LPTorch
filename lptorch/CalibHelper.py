# Some constructor may requires calibration data to be passed in, so we need to
# write a forward hook to automatically collect the data.
import torch 
import torch.nn as nn 
import uuid
# handle output in tuple, if tuple, only get the first result
def get_first_output(output):
    if type(output) == tuple:
        return output[0]
    else:
        return output



class CalibHelper:
    def __init__(self, model:nn.Module=None) -> None:
        self.forward_hook_targets = [nn.Linear, nn.Conv2d]
        self.bs = 1
        if model is not None:
            self.set_model(model)
        pass

        self.collected_calib_data = {}
        self.collected_input_shape = {}
        
        # some case calib is also not available, use fake to fake the calib data
        self.fake = False
        self.named_fake_calib_data = {}
        self.named_fake_input_shape = {}
    
    def set_model(self, model):
        self.assign_unique_id_to_all_layers(model)
        self.default_hook = self.int_scale_forward_hook
    

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
            self.collected_input_shape[unique_id] = input_record.shape
            self.collected_calib_data[unique_id] = input_record.detach().cpu() # store in cpu
        else:
            self.collected_calib_data[unique_id] = 0.9 * self.collected_calib_data[unique_id] + 0.1 * input_record.detach().cpu()
        return output

    def int_scale_forward_hook(self, module, input, output):
        unique_id = module.unique_id
        x = get_first_output(input)
        y_gt = get_first_output(output)
        x_scale = x.abs().max() / 127
        y_scale = y_gt.abs().max() / 127
        # running mean
        if unique_id not in self.collected_calib_data:
            self.collected_input_shape[unique_id] = x.shape
            self.collected_calib_data[unique_id] = [x_scale.detach().cpu(), y_scale.detach().cpu()]  # store in cpu
        else:
            self.collected_calib_data[unique_id] = [0.9 * self.collected_calib_data[unique_id][0] + 0.1 * x_scale.detach().cpu(), 
                                                    0.9 * self.collected_calib_data[unique_id][1] + 0.1 * y_scale.detach().cpu()]
        return output
    
    def register_forward_hooks(self):
        self.fwd_hooks = []
        for layer_id, layer in self.id_to_layer.items():
            if type(layer) in self.forward_hook_targets:
                hook = layer.register_forward_hook(self.default_hook)
                self.fwd_hooks.append(hook)
        self.collected_calib_data = {}
        self.collected_input_shape = {}

    def remove_forward_hooks(self):
        for hook in self.fwd_hooks:
            hook.remove()
    
    # some cases the calib of the whole model is also not available
    # but we just want to test the performance of the whole model
    # this case, we provides a dict to store fake calib_data manually.
    # e.g. collect the FFN input calib and SELFATTN calib. Then use it to broadcast to all models
    def set_fake(self):
        self.fake = True
    
    def turn_off_fake(self):
        self.fake = False
        self.named_fake_calib_data = {}
        self.named_fake_input_shape = {}

    def set_fake_module_calib_data(self, module_name, calib_input_shape, calib_result):
        assert self.fake, "only under fake mode can set calib data"
        self.named_fake_calib_data[module_name] = calib_result
        self.named_fake_input_shape[module_name] = calib_input_shape

    def save_fake_calib_data(self, path="./fake_calib_data.pkl"):
        saved_data = {
            "named_fake_calib_data": self.named_fake_calib_data,
            "named_fake_input_shape": self.named_fake_input_shape
        }
        # use pickle
        torch.save(saved_data, path)
    
    def load_fake_calib_data(self, path="./fake_calib_data.pkl"):
        saved_data = torch.load(path)
        self.named_fake_calib_data = saved_data["named_fake_calib_data"]
        self.named_fake_input_shape = saved_data["named_fake_input_shape"]

    # mannually set unique id
    def man_set_unique_id(self, module):
        # use uuid
        unique_id = str(uuid.uuid4())
        module.unique_id = unique_id
        return unique_id
    
    def man_set_module_calib_data(self, module, calib_result):
        self.collected_calib_data[module.unique_id] = calib_result
    
    '''
    get the calib data of a module=====================
    '''
    
    def get_module_calib_data(self, module):
        if not hasattr(module, 'unique_id'):
            return None
        if module.unique_id not in self.collected_calib_data:
            return None
        return self.collected_calib_data[module.unique_id]

    def set_bs(self, bs):
        self.bs = bs

    def get_module_input_shape(self, module):
        if not hasattr(module, 'unique_id'):
            return None
        if module.unique_id not in self.collected_input_shape:
            return None
        shape_input = self.collected_input_shape[module.unique_id]
        # if the first shape is 1, then multiply the first dim value with bs
        if shape_input[0] == 1:
            shape_input = [self.bs] + list(shape_input[1:])
        return shape_input

    
    def clear_calib_data(self):
        self.collected_calib_data = {}
        self.collected_input_shape = {}

    # Hanlde the distributed case
    # The calib data collection may be decoupled with the quantization. 
    def save_calib_data(self, path="./calib_data.pkl"):
        save_result = {
            "input_shape": self.collected_input_shape,
            "calib_data": self.collected_calib_data
        }
        torch.save(save_result, path)
        pass

    # load calib data from disk
    def load_calib_data(self, path="./calib_data.pkl"):
        load_result = torch.load(path)
        self.collected_input_shape = load_result["input_shape"]
        self.collected_calib_data = load_result["calib_data"]
        pass

    def set_module_calib_data_to_module(self):
        for unique_id, calib_data in self.collected_calib_data.items():
            self.id_to_layer[unique_id].calib_data = calib_data
            self.id_to_layer[unique_id].has_calib_data = True
    
    # remove
    def remove_calib_data_from_module(self):
        for unique_id, calib_data in self.collected_calib_data.items():
            self.id_to_layer[unique_id].calib_data = None
            self.id_to_layer[unique_id].has_calib_data = False
    

    
