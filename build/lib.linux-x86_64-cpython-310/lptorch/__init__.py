from . import gptq
from . import torch_int
from .extern_qlinear_constructor import construct_quantized_linear
from .AdaQLinear import (
    AdaQLinear, quantize_linear_module_with_bit, quantize_one_linear_module, ForwardTokenizer,
    AdaQTPConfig
)
from .CalibHelper import CalibHelper
from . import tp 

from . import config
from .config import is_available_q_method
from ._globals import inner_caliber

import os
# SET DEFAULT QUANTIZATION METHOD
os.environ.setdefault('Q_METHOD', 'ADA')
os.environ.setdefault('LP_BITS_THRESHOLD', '6.0')
Q_METHOD = os.environ.get('Q_METHOD')
def set_q_method(method:str):
    method = method.upper()
    is_available_q_method(method)
    os.environ['Q_METHOD'] = method
    print(f"Q_METHOD is set to {method}")
