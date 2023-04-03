from lptorch import CalibHelper, AdaQLinear
from lptorch.utils import get_capability, perf_utils, get_model_size_cuda
from lptorch import set_q_method
import torch
import torch.nn as nn 
from icecream import ic

@torch.no_grad()
def test_set_QType():
    cap = get_capability()
    # sample case
    B, M, N = 128, 512, 1024
    # sample_x, qx, and sample linear
    sample_x = torch.randn(B, M)
    linear = torch.nn.Linear(M, M, bias=True)
    linear2 = torch.nn.Linear(M, N, bias=True)
    seq_mod = nn.Sequential(linear, linear2)

    caliber = CalibHelper(seq_mod)
    # caliber.default_hook = caliber.torch_int_forward_hook
    caliber.register_forward_hooks()
    y = seq_mod(sample_x)
    caliber.remove_forward_hooks()

    linear_1 = linear.cuda()
    print(get_model_size_cuda(linear_1 , unit='MB'))

    calib_d_1 = caliber.get_module_calib_data(linear)
    x_scale, y_scale = calib_d_1
    ada_linear_1 = AdaQLinear(linear, 16, 16, x_scale=x_scale, y_scale=y_scale)
    ada_linear_1 = ada_linear_1.cuda()
    print(get_model_size_cuda(ada_linear_1 , unit='MB'))
    print(ada_linear_1.inner_layer.weight.dtype)




test_set_QType()