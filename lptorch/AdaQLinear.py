import torch 
from torch import nn 
from .utils import uniform_dtype

# implementation of self-qlinear: AdaQLinear
# the adaint constructor supposed to
# Dtype
    # 1) determine the input dtype: fp16 or int8
    # 2) also determine the output dtype: fp16 or int8
    # This allows to utilize the communication cost as well as the casting cost.
# Use different INT implementation
    # 1) Tensorcore is allowed, use cutlass
    # 2) Tensorcore is not allowed, use simple gemm
# need: intermediate dtype: force converting all the output to the dtype, by default=fp16
# to function: some int, like torch_int didn't support

'''
The Design of AdaQLinear:
User externally specify the input dtype (bit), output dtype (bit), and kernel dtype (bit)
- options: sample_input:torch.Tensor may used in torch_int implementation
Then AdaQ directly specifies 
1) the best execution kernel
2) pre_forward_quantizer and after_forward_quantizer
tokenizer could be none, once
- The input_dtype == kernel_dtype.input
- The output_dtype == kernel_dtype.output
Each tokenizer can be dispacted
- Which means it may be executed earlier than the linear kernel, or
- It may be executed later than the linear kernel
So we have two more indicators:
- pre_tokenizer_dispatch, and after_tokenizer_dispatch, default: Fasle
- once ture, the corresponding tokenizer won't be executed in the forward function
'''

from .extern_qlinear_constructor import construct_quantized_linear

# The tokenizer tries to adapt data to right precision
# FP16 -> INT8 and INT8 -> FP16
class ForwardTokenizer(nn.Module):
    def __init__(self, input_bit, output_bit, y_scale=None):
        super().__init__()
        # we only consider two cases
        if input_bit == 8 and output_bit == 16:
            assert y_scale is not None, "y_scale is required for int8 to fp16"
            self.tokenizer = self.int8_to_fp16
            self.tokenizer_type = "int8_to_fp16"
        elif input_bit == 16 and output_bit == 8:
            self.tokenizer = self.fp16_to_int8
            self.tokenizer_type = "fp16_to_int8"
        else:
            self.tokenizer = lambda x: x # empty function
            self.tokenizer_type = "empty"
        
        self.y_scale = y_scale
    
    def int8_to_fp16(self, x):
        q_y = (x * self.y_scale).to(torch.float16)
        return q_y

    def fp16_to_int8(self, x):
        x_scale = x.abs().max() / 127
        qx = (x / x_scale).round().to(torch.int8)
        return qx
    
    @torch.no_grad()
    def forward(self, x):
        if self.tokenizer:
            return self.tokenizer(x)
        else:
            return x
    
    # name
    def __repr__(self): 
        return self.tokenizer_type

# Usually, this happens after we already got the best precision for the kernels
# then we knows the former precision of the input and output
# the tokenizer alwasy works for only CUTLASS KERNELS
def construct_inner_kernel(layer:nn.Module, input_bit:int, kernel_bit:int, \
                            sample_input:torch.Tensor=None, x_scale:torch.Tensor=None, y_scale:torch.Tensor=None, \
                            device_cap=70, output_bit=16):
    after_forward_quantizer = None
    pre_forward_quantizer = None
    if device_cap <= 70 or kernel_bit < 8 or (sample_input is None and x_scale is None): # CUTLASS is not allowed / bit < 8 / no sample input
        # use GPTQ, GPTQ is a weight only quantization method, no tokenizer is needed
        inner_layer = construct_quantized_linear(layer, bit=kernel_bit, constructor='gptq')
    elif kernel_bit == 8:
        # CUTLASS OP IS AVAILABLE
        # Cal Y_SCALE
        pre_forward_quantizer = ForwardTokenizer(input_bit, 8)
        LinearType='W8A8B8O8Linear'
        if x_scale is None:
            layer, sample_input = uniform_dtype((layer, sample_input), dtype=torch.float32)
            y_gt = layer(sample_input)
            y_scale = y_gt.abs().max() / 127
            inner_layer = construct_quantized_linear(layer, kernel_bit, sample_input=sample_input, constructor='torch_int', LinearType=LinearType)
        else:
            inner_layer = construct_quantized_linear(layer, kernel_bit, x_scale=x_scale, y_scale=y_scale, constructor='torch_int', LinearType=LinearType)
        # print("Choosen LinearType: ", LinearType)
        after_forward_quantizer = ForwardTokenizer(8, output_bit, y_scale)
    else:
        # other precision is not required to do anything
        # we use fp16 by default
        inner_layer = layer.to(torch.float16)
    return inner_layer, pre_forward_quantizer, after_forward_quantizer

class AdaQLinear(nn.Module):
    def __init__(self, layer:nn.Module, input_bit:int, kernel_bit:int, \
                 sample_input:torch.Tensor=None, x_scale=None, y_scale=None, \
                 device_cap:int=70):
        super().__init__()
        self.input_bit = input_bit
        self.kernel_bit = kernel_bit

        # tokenizers
        # PS. the tokenizer may need to be iterable to the input, be careful for this case later.
        self.pre_forward_quantizer = None
        self.after_forward_quantizer = None
        # indicators
        self.pre_tokenizer_dispatch = False
        self.after_tokenizer_dispatch = False

        assert sample_input is not None or (x_scale is not None and y_scale is not None), "Either sample_input or x_scale and y_scale is required"
        if x_scale is not None and y_scale is not None:
            self.inner_layer, self.pre_forward_quantizer, self.after_forward_quantizer = \
                construct_inner_kernel(layer, input_bit, kernel_bit, x_scale=x_scale, y_scale=y_scale, device_cap=device_cap)
        if sample_input is not None:
            # make sure sample is fp16
            sample_input = sample_input.to(torch.float16)
            # construct the inner layer
            self.inner_layer, self.pre_forward_quantizer, self.after_forward_quantizer = \
                construct_inner_kernel(layer, input_bit, kernel_bit, sample_input=sample_input, device_cap=device_cap)
        
    def dispatch_pre_tokenizer(self):
        if self.pre_forward_quantizer:
            if not self.pre_tokenizer_dispatch:
                self.pre_tokenizer_dispatch = True
                return self.pre_forward_quantizer
        return False

    def forward(self, input):
        if not self.pre_tokenizer_dispatch and self.pre_forward_quantizer:
            input = self.pre_forward_quantizer(input)
        output = self.inner_layer(input)
        if not self.after_tokenizer_dispatch and self.after_forward_quantizer:
            output = self.after_forward_quantizer(output)
        return output