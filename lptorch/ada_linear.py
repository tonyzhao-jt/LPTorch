import torch 
from torch import nn 

from .gptq.nn import QuantLinear, Quantizer, quantize
from .torch_int.nn import W8A16Linear
from .torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU
import bitsandbytes as bnb
from bitsandbytes.nn.modules import Linear8bitLt

def gptq_constructor(layer:nn.Module, bit:int, sample_input:torch.Tensor=None):
    quantizer = Quantizer()
    quantizer.configure(bit, perchannel=True, sym=False, mse=False)
    quantizer.find_params(layer.weight.data, weight=True)
    layer.weight.data = quantize(
        layer.weight.data, quantizer.scale, quantizer.zero, quantizer.maxq
    )
    # group size by default = -1
    qlayer = QuantLinear(bit, -1, layer.in_features, layer.out_features)
    qlayer.pack(layer, quantizer.scale, quantizer.zero)
    return qlayer

def torch_int_constructor(layer:nn.Module, bit:int, sample_input:torch.Tensor=None, LinearType='W8A8B8O8Linear'):
    x = sample_input
    x_scale = x.abs().max() / 127
    if LinearType == 'W8A8B8O8Linear':
        QLinearType = W8A8B8O8Linear
    elif LinearType == 'W8A8B8O8LinearReLU':
        QLinearType = W8A8B8O8LinearReLU
    elif LinearType == 'W8A8BFP32OFP32Linear':
        QLinearType = W8A8BFP32OFP32Linear
    else:
        QLinearType = W8A16Linear
    q_linear = QLinearType.from_float(layer, x_scale)
    return q_linear

def bitsandbytes_consttuctor(layer:nn.Module, bit:int, sample_input:torch.Tensor=None):
    # get layer weight dim
    linear = layer
    linear_custom = Linear8bitLt(
        linear.in_features,
        linear.out_features,
        linear.bias is not None,
        has_fp16_weights=False,
        threshold=6.0,
    )
    linear_custom.state.force_no_igemmlt = True

    linear_custom.weight = bnb.nn.Int8Params(
        linear.weight.data.clone(), requires_grad=False, has_fp16_weights=False
    ).to(linear.weight.dtype)
    linear_custom.bias = linear.bias
    return linear_custom

# the adaint constructor supposed to
# Dtype
    # 1) determine the input dtype: fp16 or int8
    # 2) also determine the output dtype: fp16 or int8
    # This allows to utilize the communication cost as well as the casting cost.
# Use different INT implementation
    # 1) Tensorcore is allowed, use cutlass
    # 2) Tensorcore is not allowed, use simple gemm
def adaint_constructor(layer:nn.Module, bit:int, sample_input:torch.Tensor=None):
    # ada int suppose to support:
    return layer

def construct_quantized_linear(layer:nn.Module, bit:int, sample_input:torch.Tensor=None, constructor:str='gptq', \
                               LinearType='W8A8B8O8Linear'):
    assert isinstance(layer, nn.Linear), "Only support linear layer"
    if constructor is None or bit < 8:
        # only gptq support bit < 8
        constructor = gptq_constructor
    elif constructor == 'gptq':
        constructor = gptq_constructor
    elif constructor == 'torch_int':
        return torch_int_constructor(layer, bit, sample_input, LinearType=LinearType)
    elif constructor == 'bitsandbytes':
        constructor = bitsandbytes_consttuctor
    elif constructor == 'adaint':
        constructor = adaint_constructor
    else:
        raise ValueError("constructor must be one of gptq, torch_int, bitsandbytes")
    return constructor(layer, bit, sample_input)


# need: intermediate dtype: force converting all the output to the dtype, by default=fp16
# to function: some int, like torch_int didn't support

class AdaLinear(nn.Module):
    def __init__(self, layer:nn.Module, bit:int, sample_input:torch.Tensor=None, constructor:str='gptq', \
                  input_dtype=torch.float16, output_dtype=torch.float16):
        super().__init__()
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.bit = bit
        self.constructor = constructor
        self.inner_layer = construct_quantized_linear(layer, bit, sample_input, constructor)
        # not allowed to reset bit for the moment

    def forward(self, input):
        input = input.to(self.input_dtype)
        return self.inner_layer(input).to(self.output_dtype)