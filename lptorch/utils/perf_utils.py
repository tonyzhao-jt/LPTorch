from time import perf_counter
import torch
import copy
def time_perf_cuda():
    torch.cuda.synchronize()
    return perf_counter()

def run_on_cuda(input_x, linear, x_dtype=None, cnt_times=100):
    # deep copy input_x and linear
    input_x = copy.deepcopy(input_x)
    linear = copy.deepcopy(linear)
    input_x = input_x.cuda()
    linear = linear.cuda()
    ref_out = None
    if x_dtype is not None:
        input_x = input_x.to(x_dtype)
    # warm up
    for _ in range(10):
        ref_out = linear(input_x)
    torch.cuda.synchronize()
    # time
    start = time_perf_cuda()
    for _ in range(cnt_times):
        ref_out = linear(input_x)
    torch.cuda.synchronize()
    end = time_perf_cuda()
    return (end - start) / cnt_times, ref_out