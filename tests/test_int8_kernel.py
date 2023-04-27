from lptorch.torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU
import torch
from icecream import ic
# li =  W8A8B8O8Linear(1024, 1024)
# li = li.cuda()
# rand_x = torch.randint(-127, 127, (4, 512, 1024), dtype=torch.int8)
# rand_x = rand_x.cuda()
# res = li(rand_x)

# print(res)
# print(li.weight)


def test_w8a8b8o8_linear():
    B, M, N = 2048, 1024, 1024
    x = torch.randn(B, M)
    x_scale = x.abs().max() / 127
    qx = (x / x_scale).round().to(torch.int8)
    linear = torch.nn.Linear(M, N, bias=True)
    y_gt = linear(x)
    y_scale = y_gt.abs().max() / 127
    print(x_scale, y_scale)
    q_linear = W8A8B8O8Linear.from_float(linear, x_scale, y_scale).cuda()
    q_y = q_linear(qx.cuda()).cpu()
    y_hat = q_y * y_scale
    r2 = (y_gt - y_hat).pow(2).mean() / y_gt.pow(2).mean()
    ic(r2)

test_w8a8b8o8_linear()