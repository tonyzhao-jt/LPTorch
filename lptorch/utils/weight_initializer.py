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
                torch.randint(-1, 1, weight_shape, dtype=w_dtype, device=module.weight.device, out=module.weight)
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
                torch.randint(-1, 1, bias_shape, dtype=b_dtype, device=module.bias.device, out=module.bias)
            else:
                raise NotImplementedError


def init_weight_bias_with_rand_GPTQ(module):
    # self.register_buffer('qzeros', torch.zeros((math.ceil(infeatures/groupsize),outfeatures // 256 * (bits * 8)), dtype=torch.int))
    # self.register_buffer('scales', torch.zeros((math.ceil(infeatures/groupsize),outfeatures)))
    # self.register_buffer('bias', toårch.zeros(outfeatures))
    # self.register_buffer(
    #     'qweight', torch.zeros((infeatures // 32 * bits, outfeatures), dtype=torch.int)
    # )
    # gptq has qweight, qzeros, scales, bias
    weight_shape = module.qweight.shape
    w_dtype = module.qweight.dtype
    scales_type = module.scales.dtype
    bias_type = module.bias.dtype
    qzeros_type = module.qzeros.dtype
    torch.randint(-2145032089, 2142485127, weight_shape, dtype=w_dtype, device=module.qweight.device, out=module.qweight)
    # torch.randint(-127, 127, module.qzeros.shape, dtype=w_dtype, device=module.qzeros.device, out=module.qzeros)
    module.scales.uniform_(0.0126, 0.0940)
    module.bias.uniform_(-0.6216, 0.5781)
    torch.randint(-2056878265,2021095270, module.qzeros.shape, dtype=qzeros_type, device=module.qzeros.device, out=module.qzeros)

