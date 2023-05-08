import torch 
from torch import nn 
import torch.distributed as dist
import os

from .utils import uniform_dtype, get_capability
from .config import is_available_bit
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
            self.y_scale = nn.Parameter(torch.tensor(y_scale.numpy(), dtype=torch.float16))
        elif input_bit == 16 and output_bit == 8:
            self.tokenizer = self.fp16_to_int8
            self.tokenizer_type = "fp16_to_int8"
            self.y_scale = None
        else:
            # if output_bit == 32:
            #     self.tokenizer = lambda x: x.float()
            if input_bit == output_bit and input_bit == 8:
                # self.tokenizer = lambda x: x # empty function
                self.tokenizer = self.empty_fwd
                self.tokenizer_type = "empty"
                self.y_scale = None
            else:
                # force converting to FP16. This is the default behavior
                # self.tokenizer = lambda x: x.to(torch.float16) if x.is_cuda else x # convert input tensor to fp16 if on CUDA device
                self.tokenizer = self.to_fp16
                self.tokenizer_type = "to_fp16"
                self.y_scale = None
    
    def empty_fwd(self, x):
        return x 

    @torch.no_grad()
    def to_fp16(self, x):
        return x.to(torch.float16)
    
    @torch.no_grad()
    def int8_to_fp16(self, x):
        x = x.to(torch.float16)
        q_y = x * self.y_scale
        return q_y

    @torch.no_grad()
    def fp16_to_int8(self, t):
        scale = t.abs().max() / 127
        if not t.is_cuda:
            # half rounding is not supported on CPU
            t = t.float()
        # use inplace operation to save memory
        t.div_(scale).clamp_(-127, 127).round_()
        t_q = t.to(torch.int8)
        # return t_q, scale
        return t_q # TODO: make lptorch support scale later.
    
    @torch.no_grad()
    def forward(self, x):
        if self.tokenizer:
            return self.tokenizer(x)
        else:
            return x
    
    # name
    def __repr__(self): 
        return self.tokenizer_type

def construct_torch_int(layer:nn.Module, input_bit:int, kernel_bit:int, \
                            sample_input:torch.Tensor=None, x_scale:torch.Tensor=None, y_scale:torch.Tensor=None, \
                            output_bit=16, LinearType='W8A8B8O8Linear'):
    # Cal Y_SCALE
    pre_forward_quantizer = ForwardTokenizer(input_bit, 8)
    if x_scale is None:
        layer, sample_input = uniform_dtype((layer, sample_input), dtype=torch.float32)
        y_gt = layer(sample_input)
        y_scale = y_gt.abs().max() / 127
        inner_layer = construct_quantized_linear(layer, kernel_bit, sample_input=sample_input, constructor='torch_int', LinearType=LinearType)
    else:
        inner_layer = construct_quantized_linear(layer, kernel_bit, x_scale=x_scale, y_scale=y_scale, constructor='torch_int', LinearType=LinearType)
    # print("Choosen LinearType: ", LinearType)
    after_forward_quantizer = ForwardTokenizer(8, output_bit, y_scale)
    return inner_layer, pre_forward_quantizer, after_forward_quantizer

# Usually, this happens after we already got the best precision for the kernels
# then we knows the former precision of the input and output
# the tokenizer alwasy works for only CUTLASS KERNELS
def construct_inner_kernel(layer:nn.Module, input_bit:int, kernel_bit:int, \
                            sample_input:torch.Tensor=None, x_scale:torch.Tensor=None, y_scale:torch.Tensor=None, \
                            device_cap=70, output_bit=16, LinearType='W8A8B8O8Linear'):
    layer_type = 'FP16'
    Q_METHOD = os.environ.get('Q_METHOD')
    # print("Q_METHOD: ", Q_METHOD, kernel_bit)
    if Q_METHOD == 'GPTQ':
        inner_layer = construct_quantized_linear(layer, bit=kernel_bit, constructor='gptq')
        layer_type = 'GPTQ'
        return inner_layer, None, None, layer_type
    elif Q_METHOD == 'TORCH_INT':
        if device_cap <= 70:
            # can't use torch_int
            return layer, None, None, layer_type
        # same to ADA
        layer_type = 'TORCH_INT:' + LinearType
        inner_layer, pre_forward_quantizer, after_forward_quantizer = construct_torch_int(layer, input_bit, kernel_bit, \
                                                                                            sample_input, x_scale, y_scale, output_bit,\
                                                                                            LinearType=LinearType)
        return inner_layer, pre_forward_quantizer, after_forward_quantizer, layer_type
    elif Q_METHOD == 'BITSANDBYTES':
        layer_type = 'BITSANDBYTES'
        inner_layer = construct_quantized_linear(layer, kernel_bit, sample_input=sample_input, constructor='bitsandbytes')
        return inner_layer, None, None, layer_type
    else:
        # ADALINEAR
        after_forward_quantizer = None
        pre_forward_quantizer = ForwardTokenizer(input_bit, 16, y_scale=y_scale) # always convert to fp16
        if device_cap <= 70 or kernel_bit < 8 or (sample_input is None and x_scale is None): # CUTLASS is not allowed / bit < 8 / no sample input
            # use GPTQ, GPTQ is a weight only quantization method, no tokenizer is needed
            layer_type = 'GPTQ'
            inner_layer = construct_quantized_linear(layer, bit=kernel_bit, constructor='gptq')
        elif kernel_bit == 8:
            # if input bit =16 use bitsandsbytes
            if input_bit == 16:
                layer_type = 'BITSANDBYTES'
                inner_layer = construct_quantized_linear(layer, kernel_bit, sample_input=sample_input, constructor='bitsandbytes')
            else:
                # use torch int
                layer_type = 'TORCH_INT:' + LinearType
                inner_layer, pre_forward_quantizer, after_forward_quantizer = construct_torch_int(layer, input_bit, kernel_bit, \
                                                                                            sample_input, x_scale, y_scale, \
                                                                                            output_bit, LinearType=LinearType)
        else:
            # other precision is not required to do anything
            # we use fp16 by default
            inner_layer = layer.to(torch.float16)
        return inner_layer, pre_forward_quantizer, after_forward_quantizer, layer_type

# tp related
from . import tp 
import dataclasses
@dataclasses.dataclass
class AdaQTPConfig:
    split_k: int
    global_rank: int
    tp_index: int
    split_type: str
    comm_group = None

    def __init__(self, split_k: int, global_rank:int, tp_index: int, split_type: str = "COLUMN", comm_group = None):
        self.split_k = split_k
        self.global_rank = global_rank
        self.tp_index = tp_index
        self.split_type = split_type
        self.comm_group = comm_group


from .utils import partition_a_into_b_bins
class AdaQLinear(nn.Module):
    def __init__(self, layer:nn.Module, input_bit:int, kernel_bit:int,\
                 sample_input:torch.Tensor=None, x_scale=None, y_scale=None,
                 output_bit:int=16, LinearType='W8A8B8O8Linear', **kwargs):
        super().__init__()
        is_available_bit(kernel_bit)
        device_cap = get_capability()
        self.input_bit = input_bit
        self.kernel_bit = kernel_bit

        # flag paramters. help judge device and dtype
        self.flag = nn.Parameter(torch.zeros(1), requires_grad=False)

        # tokenizers
        # PS. the tokenizer may need to be iterable to the input, be careful for this case later.
        self.pre_forward_quantizer = None
        self.after_forward_quantizer = None
        # indicators
        self.pre_tokenizer_dispatch = False
        self.after_tokenizer_dispatch = False

        # add shard support here
        tp_config = kwargs.get('tp_config', None)
        if tp_config is not None:
            layer = self.shard_layer(layer, tp_config)

        layer_type = 'FP16'
        # deal with the case > 8 bit
        # if kernel_bit == 32:
        #     layer_type = 'FP32'
        #     # do nothing
        #     inner_layer = layer.to(torch.float32)
        #     pre_forward_quantizer = ForwardTokenizer(input_bit, 32)
        #     # after_forward_quantizer = ForwardTokenizer(32, 32)
        #     self.inner_layer, self.pre_forward_quantizer, self.after_forward_quantizer = inner_layer, pre_forward_quantizer, None
        if kernel_bit >= 16:
            inner_layer = layer.to(torch.float16) # use fp16 by default
            pre_forward_quantizer = ForwardTokenizer(input_bit, 16)
            # after_forward_quantizer = ForwardTokenizer(16, 16)
            self.inner_layer, self.pre_forward_quantizer, self.after_forward_quantizer = inner_layer, pre_forward_quantizer, None
        else:
            if sample_input is not None:
                # make sure sample is fp16
                sample_input = sample_input.to(torch.float16)
                # construct the inner layer
            self.inner_layer, self.pre_forward_quantizer, self.after_forward_quantizer, layer_type = \
                construct_inner_kernel(layer, input_bit, kernel_bit, sample_input=sample_input, x_scale=x_scale, \
                                        y_scale=y_scale, device_cap=device_cap, output_bit=output_bit, LinearType=LinearType)
        
        self.layer_type = layer_type
    
    # Used in TP: Load weight later.
    # COLUMN: shard_weight by the second dimension. 
    # ROW: shard_weight by the first dimension.
    @torch.no_grad()
    def shard_layer(self, layer:nn.Linear, tp_config: AdaQTPConfig):
        k = tp_config.split_k
        global_rank = tp_config.global_rank
        idx = tp_config.tp_index
        shard_type = tp_config.split_type
        comm_group = tp_config.comm_group
        # shard the layer then return. 
        in_features, out_features = layer.in_features, layer.out_features
        if comm_group is not None:
            layer = layer.cuda()
        # weight is stored as out_features, in_features. (Transpose case)
        if shard_type == 'COLUMN':
            # shard by the second dimension
            shard_size = tp.divide(out_features, k)
            # create new linear layer
            new_layer = nn.Linear(in_features, shard_size)
            bias = layer.bias
            if comm_group is None:
                # copy weight, bias
                if idx == k - 1:
                    new_layer.weight.data.copy_(layer.weight[idx*shard_size:, :])
                    if bias is not None:
                        new_layer.bias.data.copy_(layer.bias[idx*shard_size:])
                else:
                    new_layer.weight.data.copy_(layer.weight[idx*shard_size:(idx+1)*shard_size, :])
                    if bias is not None:
                        new_layer.bias.data.copy_(layer.bias[idx*shard_size:(idx+1)*shard_size])
            else:
                # communicate the weight
                weight = layer.weight.detach()
                weight = tp._scatter_first_dim(weight, global_rank, idx, k, group=comm_group)
                dist.barrier(group=comm_group)
                new_layer.weight.data.copy_(weight.detach().cpu())

                if bias is not None:
                    bias = layer.bias.detach()
                    bias = tp._scatter_first_dim(bias, global_rank, idx, k, group=comm_group)
                    dist.barrier(group=comm_group)
                    new_layer.bias.data.copy_(bias.detach().cpu())

        elif shard_type == 'ROW':
            # shard by the first dimension
            shard_size = tp.divide(in_features, k) 
            # create new linear layer
            new_layer = nn.Linear(shard_size, out_features)
            bias = layer.bias
            # copy weight, bias
            if comm_group is None:
                if idx == k - 1:
                    new_layer.weight.data.copy_(layer.weight[:, idx*shard_size:])
                    if bias is not None:
                        new_layer.bias.data.copy_(layer.bias)
                else:
                    new_layer.weight.data.copy_(layer.weight[:, idx*shard_size:(idx+1)*shard_size])
                    if bias is not None:
                        new_layer.bias.data.copy_(layer.bias)
            else:
                # communicate the weight
                weight = layer.weight.detach()
                weight = tp._scatter_last_dim(weight, global_rank, idx, k, comm_group)
                dist.barrier(group=comm_group)
                new_layer.weight.data.copy_(weight.detach().cpu())
                if bias is not None:
                    bias = layer.bias.detach() / k
                    tp._broad_cast(bias, global_rank, idx, comm_group)
                    new_layer.bias.data.copy_(bias.detach().cpu())
                dist.barrier(group=comm_group)
        
        return new_layer
    
    def print_status(self):
        # if layer name if has
        if hasattr(self, 'name'):
            print("Layer Name: ", self.name)
        print("Layer Type: ", self.layer_type)
        print("Pre Tokenizer: ", self.pre_forward_quantizer)
        print("After Tokenizer: ", self.after_forward_quantizer)
        
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

def quantize_one_linear_module(child, input_bit=16, kernel_bit=8, caliber=None, name="ada_linear", output_bit:int=16, LinearType='W8A8B8O8Linear', **kwargs):
    assert isinstance(child, nn.Linear), "Only support linear layer" 
    x_scale, y_scale = None, None
    sample_input = None
    layer_name = name
    if caliber is not None:
        calib_d_1 = caliber.get_module_calib_data(child)
        if calib_d_1 is not None:
            if type(calib_d_1) == list and len(calib_d_1) == 2:
                x_scale, y_scale = calib_d_1
            else:
                sample_input = calib_d_1
            layer_name = child.unique_id
    ada_qli = AdaQLinear(child, input_bit, kernel_bit, sample_input=sample_input, x_scale=x_scale, y_scale=y_scale, output_bit=output_bit, LinearType=LinearType, **kwargs)
    ada_qli.name = layer_name
    return ada_qli

# iteratively quantize all linear in the module
def quantize_linear_module_with_bit(module, kernel_bit=8, caliber=None, input_bit=16, **kwargs):
    def quantize_linear_inner(module):
        for name, child in module.named_children():
            if isinstance(child, AdaQLinear):
                continue
            if isinstance(child, nn.Linear):
                ada_qli = quantize_one_linear_module(child, input_bit, kernel_bit, caliber, name=name, **kwargs)
                setattr(module, name, ada_qli)
            quantize_linear_inner(child)
    quantize_linear_inner(module)

def print_all_qli_status_in_module(module):
    def print_qli_status_inner(module):
        for name, child in module.named_children():
            if isinstance(child, AdaQLinear):
                child.print_status()
            print_qli_status_inner(child)
    print_qli_status_inner(module)