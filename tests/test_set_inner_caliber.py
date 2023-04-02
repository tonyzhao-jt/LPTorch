from lptorch import quantize_linear_module_with_bit
from lptorch.AdaQLinear import print_all_qli_status_in_module
import lptorch
import torch
import torch.nn as nn 
from icecream import ic
import copy 
@torch.no_grad()
def test_set_module_bit():
    # sample case
    B, M, N = 128, 512, 1024
    # sample_x, qx, and sample linear
    sample_x = torch.randn(B, M)
    seq_model = nn.Sequential(
        nn.Linear(M, M, bias=True),
        nn.Linear(M, N, bias=True),
        nn.Linear(N, N, bias=True),
    )
    caliber = lptorch.inner_caliber
    caliber.set_model(seq_model)
    # caliber.default_hook = caliber.torch_int_forward_hook
    caliber.register_forward_hooks()
    seq_model(sample_x)
    caliber.remove_forward_hooks()

    seq_model_cp = copy.deepcopy(seq_model)
    quantize_linear_module_with_bit(seq_model_cp, kernel_bit=16)
    print_all_qli_status_in_module(seq_model_cp)

    seq_model_cp = copy.deepcopy(seq_model)
    quantize_linear_module_with_bit(seq_model_cp, kernel_bit=8)
    print_all_qli_status_in_module(seq_model_cp)

    # first try quant without calib data
    quantize_linear_module_with_bit(seq_model, kernel_bit=8, caliber=lptorch.inner_caliber)
    print_all_qli_status_in_module(seq_model)


if __name__ == "__main__":
    test_set_module_bit()