from .quantizer import Quantizer, quantize
from .linear import QuantLinear

def make_quant(module, names, bits, groupsize, name=''):
    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            delattr(module, attr)
            setattr(
                module, attr, QuantLinear(bits, groupsize, tmp.in_features, tmp.out_features)
            )
    for name1, child in module.named_children():
        make_quant(child, names, bits, groupsize, name + '.' + name1 if name != '' else name1)