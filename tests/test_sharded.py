# test the sharded result.
from lptorch import CalibHelper, quantize_linear_module_with_bit, AdaQTPConfig
from lptorch.utils import get_capability, perf_utils
from lptorch import set_q_method
import torch
import torch.nn as nn 
from icecream import ic
import copy

@torch.no_grad()
def test_set_QType_sharded():
    cap = get_capability()
    # sample case
    B, M, N = 128, 512, 1024
    # sample_x, qx, and sample linear
    sample_x = torch.randn(B, M)
    linear = torch.nn.Linear(M, M, bias=True)
    seq_mod_ = nn.Sequential(linear)

    caliber = CalibHelper(seq_mod_)
    # caliber.default_hook = caliber.torch_int_forward_hook
    caliber.register_forward_hooks()
    y = seq_mod_(sample_x)
    caliber.remove_forward_hooks()

    calib_d_1 = caliber.get_module_calib_data(linear)
    x_scale, y_scale = calib_d_1

    # simple case
    tp_config = None
    seq_mod = copy.deepcopy(seq_mod_)
    quantize_linear_module_with_bit(seq_mod, kernel_bit=8, caliber=caliber, tp_config=tp_config)
    sample_x = sample_x.cuda()
    seq_mod = seq_mod.cuda()
    print(seq_mod(sample_x).shape)

    # column wise case
    tp_config = AdaQTPConfig(split_k=2, rank_index=0, split_type='COLUMN')
    seq_mod = copy.deepcopy(seq_mod_)
    quantize_linear_module_with_bit(seq_mod, kernel_bit=8, caliber=caliber, tp_config=tp_config)
    sample_x = sample_x.cuda()
    seq_mod = seq_mod.cuda()
    print(seq_mod(sample_x).shape)

    # partitiom sample_x along the second dimension
    sample_x = sample_x.cuda()
    sample_x = sample_x.chunk(tp_config.split_k, dim=1)[tp_config.rank_index]
    tp_config = AdaQTPConfig(split_k=2, rank_index=0, split_type='ROW')
    seq_mod = copy.deepcopy(seq_mod_)
    quantize_linear_module_with_bit(seq_mod, kernel_bit=8, caliber=caliber, tp_config=tp_config)
    seq_mod = seq_mod.cuda()
    print(seq_mod(sample_x).shape)


if __name__ == "__main__":
    test_set_QType_sharded()