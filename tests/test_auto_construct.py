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
    caliber.register_forward_hooks()
    y = seq_mod(sample_x)
    caliber.remove_forward_hooks()

    calib_d_1 = caliber.get_module_calib_data(linear)
    ada_linear_1 = AdaQLinear(linear, 16, 8, calib_d_1, cap)
    # test
    y = ada_linear_1(calib_d_1)

    calib_d_2 = caliber.get_module_calib_data(linear2)
    ada_linear_2 = AdaQLinear(linear2, 16, 8, calib_d_2, cap)

    caliber.set_module_calib_data_to_module()
    print(linear.has_calib_data)
    print(linear.calib_data)

    print(linear2.has_calib_data)
    print(linear2.calib_data)

if __name__ == "__main__":
    test_calib()