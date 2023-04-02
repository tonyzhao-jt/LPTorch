from lptorch import CalibHelper, AdaQLinear
from lptorch.utils import get_capability
import torch
import torch.nn as nn 

@torch.no_grad()
def test_calib():
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

    calib_d_1 = caliber.get_module_calib_data(linear)
    x_scale, y_scale = calib_d_1
    ada_linear_1 = AdaQLinear(linear, 16, 8, x_scale=x_scale, y_scale=y_scale, device_cap=cap)
    # ada_linear_1 = AdaQLinear(linear, 16, 8, sample_input=calib_d_1, device_cap=cap)

    # not that must run on cuda
    ada_linear_1_cuda = ada_linear_1.cuda()
    sample_x_cuda = sample_x.cuda()
    y = ada_linear_1_cuda(sample_x_cuda)


    calib_d_2 = caliber.get_module_calib_data(linear2)
    x_scale, y_scale = calib_d_2
    ada_linear_2 = AdaQLinear(linear2, 16, 8, x_scale=x_scale, y_scale=y_scale, device_cap=cap)

    # not that must run on cuda
    ada_linear_2_cuda = ada_linear_2.cuda()
    y = ada_linear_2_cuda(y)

    caliber.save_calib_data()
    caliber.clear_calib_data()
    caliber.load_calib_data()
    caliber.set_module_calib_data_to_module()
    print(linear.has_calib_data)
    print(linear.calib_data)

    print(linear2.has_calib_data)
    print(linear2.calib_data)


if __name__ == "__main__":
    test_calib()