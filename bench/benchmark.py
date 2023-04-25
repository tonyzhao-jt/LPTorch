import torch 
from lptorch import construct_quantized_linear
from lptorch.utils import get_capability, perf_utils
from lptorch import AdaQLinear
from icecream import ic

@torch.no_grad()
def test_ada_linear():
    # sample case
    B, M, N = 2048, 1024, 1024
    # sample_x, qx, and sample linear
    sample_x = torch.randn(B, M)
    x_scale = sample_x.abs().max() / 127
    linear = torch.nn.Linear(M, N, bias=True)
    qx = (sample_x / x_scale).round().to(torch.int8)
    # sample output
    y_gt = linear(sample_x)
    y_scale = y_gt.abs().max() / 127

    # only INT8 this case
    bit = 8
    # test different cases
    # gptq
    gptq_linear = construct_quantized_linear(linear, bit, constructor='gptq')
    gptq_time, ref_out = perf_utils.run_on_cuda(sample_x, gptq_linear, x_dtype=torch.float16)
    ic(gptq_time)
    ic(ref_out.dtype)
    # bitsandbytes
    bitsandbytes_linear = construct_quantized_linear(linear, bit, constructor='bitsandbytes')
    bitsandbytes_time, ref_out = perf_utils.run_on_cuda(sample_x, bitsandbytes_linear, x_dtype=torch.float16)
    ic(bitsandbytes_time)
    ic(ref_out.dtype)
    # torch_int
    # For torch_int, it has multiple types
    # including: W8A8B8O8Linear, W8A8B8O8LinearReLU, W8A8BFP32OFP32Linear, W8A16Linear
    # even inlcuding BMM, for our case, we utilize memory, BMM only used when two output is both INT8 (precision reduction)
    cap = get_capability()
    if cap >= 75:
        # CUTLASS SUPPORT ONLY AMPHERE AND TURING
        torch_int_linear = construct_quantized_linear(linear, bit, sample_input=sample_x, constructor='torch_int', LinearType='W8A8B8O8Linear')
        torch_int_time_888, ref_out = perf_utils.run_on_cuda(qx, torch_int_linear, x_dtype=torch.int8)
        ic(torch_int_time_888)
        ic(ref_out.dtype)
        # 8832
        torch_int_linear = construct_quantized_linear(linear, bit, sample_input=sample_x, constructor='torch_int', LinearType='W8A8BFP32OFP32Linear')
        torch_int_time_8832, ref_out = perf_utils.run_on_cuda(qx, torch_int_linear, x_dtype=torch.int8)
        ic(torch_int_time_8832)
        ic(ref_out.dtype)
        # 81616
        torch_int_linear = construct_quantized_linear(linear, bit, sample_input=sample_x, constructor='torch_int', LinearType='W8A16Linear')
        torch_int_time_81616, ref_out = perf_utils.run_on_cuda(qx, torch_int_linear, x_dtype=torch.float16)
        ic(torch_int_time_81616)
        ic(ref_out.dtype)

    # AdaQLinear
    input_bit = 8
    kernel_bit = 8
    ada_linear = AdaQLinear(linear, input_bit, kernel_bit, sample_input=sample_x, y_scale=y_scale)
    # print(ada_linear.pre_forward_quantizer if ada_linear.pre_forward_quantizer is not None else 'None')
    ada_linear_time_8, ref_out = perf_utils.run_on_cuda(qx, ada_linear, x_dtype=torch.int8)
    ic(ada_linear_time_8)
    ic(ref_out.dtype)
    
    input_bit = 16
    ada_linear = AdaQLinear(linear, input_bit, kernel_bit, sample_input=sample_x)
    ada_linear_time_16, ref_out = perf_utils.run_on_cuda(sample_x, ada_linear, x_dtype=torch.float16)
    ic(ada_linear_time_16)
    ic(ref_out.dtype)

    input_bit = 16
    kernel_bit = 16
    ada_linear = AdaQLinear(linear, input_bit, kernel_bit, sample_input=sample_x)
    ada_linear_time_1616, ref_out = perf_utils.run_on_cuda(sample_x, ada_linear, x_dtype=torch.float16)
    ic(ada_linear_time_1616)
    ic(ref_out.dtype)

@torch.no_grad()
def test_quantizer_dispatcher():
    B, M, N = 128, 512, 1024
    # sample_x, qx, and sample linear
    sample_x = torch.randn(B, M)
    x_scale = sample_x.abs().max() / 127
    linear = torch.nn.Linear(M, N, bias=True)
    qx = (sample_x / x_scale).round().to(torch.int8)
    # sample output
    y_gt = linear(sample_x)
    cap = get_capability()

    # AdaQLinear
    input_bit = 16
    kernel_bit = 8
    ada_linear = AdaQLinear(linear, input_bit, kernel_bit, sample_x, cap)
    pre_tokenizer = ada_linear.dispatch_pre_tokenizer()
    # print(ada_linear.pre_forward_quantizer if ada_linear.pre_forward_quantizer is not None else 'None')
    ada_linear_time_8, ref_out = perf_utils.run_on_cuda(pre_tokenizer(sample_x), ada_linear, x_dtype=torch.int8)
    ic(ada_linear_time_8)
    ic(ref_out.dtype)

@torch.no_grad()
def test_different_gptq():
    # sample case
    B, M, N = 128, 512, 1024
    # sample_x, qx, and sample linear
    sample_x = torch.randn(B, M)
    x_scale = sample_x.abs().max() / 127
    linear = torch.nn.Linear(M, N, bias=True)
    qx = (sample_x / x_scale).round().to(torch.int8)
    # sample output
    # only INT8 this case
    bits = [2,3,4,8]
    # test different cases
    # gptq
    for bit in bits:
        gptq_linear = construct_quantized_linear(linear, bit, sample_input=sample_x, constructor='gptq')
        gptq_time, ref_out = perf_utils.run_on_cuda(sample_x, gptq_linear, x_dtype=torch.float16)
        ic(gptq_time)

if __name__ == '__main__':
    torch.cuda.set_device(0) # set device to invoke cudaSetDevice(device_id) make sure the workspace is well allocated.
    test_ada_linear()
    test_different_gptq()
    # test_quantizer_dispatcher()