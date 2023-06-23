import torch
import torch.nn as nn

from .gptq.nn import QuantLinear, Quantizer, quantize
from .torch_int.nn import W8A16Linear
from .torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU

# handle T4. in the future, fully replace the torchint
from .lptorch.nn import W8A16Linear as LPW8A16Linear
from .lptorch.nn.linear import W8A8BFP32OFP32Linear as LPW8A8BFP32OFP32Linear
from .lptorch.nn.linear import W8A8B8O8Linear as LPW8A8B8O8Linear
from .lptorch.nn.linear import W8A8B8O8LinearReLU as LPW8A8B8O8LinearReLU

import bitsandbytes as bnb
from bitsandbytes.nn.modules import Linear8bitLt

from .utils import uniform_dtype, get_capability
from .config import is_available_bit
import os 
from .utils import init_weight_bias_with_rand, init_weight_bias_with_rand_GPTQ
def gptq_constructor(layer:nn.Module, bit:int, sample_input:torch.Tensor=None):
    perf_mode = os.environ.get('PERF_MODE', False) == "1"
    qlayer = QuantLinear(bit, -1, layer.in_features, layer.out_features)
    if perf_mode:
        init_weight_bias_with_rand_GPTQ(qlayer)
        return qlayer
    quantizer = Quantizer()
    quantizer.configure(bit, perchannel=True, sym=False, mse=False)
    quantizer.find_params(layer.weight.data, weight=True)
    layer.weight.data = quantize(
        layer.weight.data, quantizer.scale, quantizer.zero, quantizer.maxq
    )
    # group size by default = -1
    qlayer.pack(layer, quantizer.scale, quantizer.zero)
    return qlayer


def torch_int_constructor_withscale(layer:nn.Module, bit:int, x_scale, y_scale, LinearType='W8A8B8O8Linear'):
    cap = get_capability()
    perf_mode = os.environ.get('PERF_MODE', False) == "1"
    if not perf_mode:
        if LinearType == 'W8A8B8O8Linear':
            QLinearType = W8A8B8O8Linear if cap > 75 else LPW8A8B8O8Linear
            q_linear = QLinearType.from_float(layer, x_scale, y_scale)
        elif LinearType == 'W8A8B8O8LinearReLU':
            QLinearType = W8A8B8O8LinearReLU if cap > 75 else LPW8A8B8O8LinearReLU
            q_linear = QLinearType.from_float(layer, x_scale, y_scale) 
        elif LinearType == 'W8A8BFP32OFP32Linear':
            QLinearType = W8A8BFP32OFP32Linear if cap > 75 else LPW8A8BFP32OFP32Linear
            q_linear = QLinearType.from_float(layer, x_scale)
        elif LinearType == 'W8A16Linear':
            QLinearType = W8A16Linear if cap > 75 else LPW8A16Linear
            q_linear = QLinearType.from_float(layer)
        else:
            q_linear = layer # didn't do anything
    else:
        in_features, out_features = layer.in_features, layer.out_features
        if LinearType == 'W8A8B8O8Linear':
            QLinearType = W8A8B8O8Linear if cap > 75 else LPW8A8B8O8Linear
            q_linear = QLinearType(in_features, out_features)
            init_weight_bias_with_rand(q_linear)
        elif LinearType == 'W8A8B8O8LinearReLU':
            QLinearType = W8A8B8O8LinearReLU if cap > 75 else LPW8A8B8O8LinearReLU
            q_linear = QLinearType(in_features, out_features)
            init_weight_bias_with_rand(q_linear)
        elif LinearType == 'W8A8BFP32OFP32Linear':
            QLinearType = W8A8BFP32OFP32Linear if cap > 75 else LPW8A8BFP32OFP32Linear
            q_linear = QLinearType(in_features, out_features)
            init_weight_bias_with_rand(q_linear)
        elif LinearType == 'W8A16Linear':
            QLinearType = W8A16Linear if cap > 75 else LPW8A16Linear
            q_linear = QLinearType(in_features, out_features)
            init_weight_bias_with_rand(q_linear)
        else:
            q_linear = layer # didn't do anything
            
    return q_linear

def torch_int_constructor(layer:nn.Module, bit:int, sample_input:torch.Tensor=None, LinearType='W8A8B8O8Linear'):
    layer, sample_input = uniform_dtype((layer, sample_input), dtype=torch.float32)
    x = sample_input
    x_scale = x.abs().max() / 127
    y_gt = layer(sample_input)
    y_scale = y_gt.abs().max() / 127
    return torch_int_constructor_withscale(layer, bit, x_scale, y_scale, LinearType=LinearType)


def bitsandbytes_consttuctor(layer:nn.Module, bit:int, sample_input:torch.Tensor=None):
    threshold = float(os.environ.get('LP_BITS_THRESHOLD', '6.0'))
    # get layer weight dim
    linear = layer
    linear_custom = Linear8bitLt(
        linear.in_features,
        linear.out_features,
        linear.bias is not None,
        has_fp16_weights=False,
        threshold=threshold, # by default set 6.0, but is weight-only quantization.
    )
    # linear_custom.state.force_no_igemmlt = True # 6.23: don't turn off the igemmlt, elsewise the performance will be bad.(easily cause nan)
    # linear_custom.state.force_no_igemmlt = False 
    linear_custom.weight = bnb.nn.Int8Params(
        data=linear.weight.data.clone(), requires_grad=False, has_fp16_weights=False
    ).to(linear.weight.dtype)
    # linear_custom.weight.cuda(torch.cuda.current_device())
    linear_custom.bias = linear.bias
    return linear_custom



def construct_quantized_linear(layer:nn.Module, bit:int, constructor:str='gptq', \
                               sample_input:torch.Tensor=None, x_scale:torch.Tensor=None, y_scale:torch.Tensor=None, LinearType='W8A8B8O8Linear'):
    assert isinstance(layer, nn.Linear), "Only support linear layer"
    is_available_bit(bit)
    if constructor is None or bit < 8:
        # only gptq support bit < 8
        constructor = gptq_constructor
    elif constructor == 'gptq':
        constructor = gptq_constructor
    elif constructor == 'torch_int':
        assert sample_input is not None or (x_scale is not None and y_scale is not None), "Either sample_input or x_scale and y_scale is required for torch_int"
        if x_scale is not None and y_scale is not None:
            return torch_int_constructor_withscale(layer, bit, x_scale, y_scale, LinearType=LinearType)
        return torch_int_constructor(layer, bit, sample_input, LinearType=LinearType)
    elif constructor == 'bitsandbytes':
        constructor = bitsandbytes_consttuctor
    else:
        raise ValueError("constructor must be one of gptq, torch_int, bitsandbytes")
    return constructor(layer, bit, sample_input)
