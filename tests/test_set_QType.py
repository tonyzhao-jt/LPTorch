from lptorch import CalibHelper, AdaQLinear
from lptorch.utils import get_capability, perf_utils
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

    calib_d_1 = caliber.get_module_calib_data(linear)
    x_scale, y_scale = calib_d_1
    ada_linear_1 = AdaQLinear(linear, 16, 8, x_scale=x_scale, y_scale=y_scale, device_cap=cap)

    time_ada_1, ref_out = perf_utils.run_on_cuda(sample_x, ada_linear_1, x_dtype=torch.float16)
    ic(time_ada_1)
    ic(ada_linear_1.layer_type)

    set_q_method('GPTQ')
    ada_linear_2 = AdaQLinear(linear, 16, 4, x_scale=x_scale, y_scale=y_scale, device_cap=cap)
    time_ada_2, ref_out1 = perf_utils.run_on_cuda(sample_x, ada_linear_2, x_dtype=torch.float16)
    ic(time_ada_2)
    ic(ada_linear_2.layer_type)
    ic(ada_linear_2.inner_layer.bits)
    print(ada_linear_2.inner_layer.qweight.shape)

    ada_linear_3 = AdaQLinear(linear, 16, 8, x_scale=x_scale, y_scale=y_scale, device_cap=cap)
    time_ada_3, ref_out2 = perf_utils.run_on_cuda(sample_x, ada_linear_3, x_dtype=torch.float32)
    # time_ada_3, ref_out2 = perf_utils.run_on_cuda(sample_x, ada_linear_3, x_dtype=torch.float16)
    ic(time_ada_3)
    ic(ada_linear_3.layer_type)
    ic(ada_linear_3.inner_layer.bits)
    print(ada_linear_3.inner_layer.qweight.shape)
    print(torch.max(ref_out2 - ref_out1))
    print(torch.max(ref_out2 - ref_out))
    


if __name__ == "__main__":
    test_set_QType()