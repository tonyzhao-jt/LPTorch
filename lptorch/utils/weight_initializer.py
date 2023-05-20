import torch
def init_weight_bias_with_rand(module):
    if hasattr(module, 'weight'):
        weight_shape = module.weight.shape
        w_dtype = module.weight.dtype
        if w_dtype == torch.float32 or w_dtype == torch.float16:
            torch.nn.init.uniform_(module.weight)
        else:
            # int8
            if w_dtype == torch.int8:
                torch.randint(-8, 8, weight_shape, dtype=w_dtype, device=module.weight.device, out=module.weight)
            else:
                raise NotImplementedError
    
    if hasattr(module, 'bias'):
        # first check if it is none
        if module.bias is None:
            return
        bias_shape = module.bias.shape
        b_dtype = module.bias.dtype
        if b_dtype == torch.float32 or b_dtype == torch.float16:
            torch.nn.init.uniform_(module.bias)
        else:
            # int8
            if b_dtype == torch.int8:
                torch.randint(-8, 8, bias_shape, dtype=b_dtype, device=module.bias.device, out=module.bias)
            else:
                raise NotImplementedError