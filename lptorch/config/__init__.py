AVAILABLE_BITS = [2, 3, 4, 8, 16]
AVAILABLE_Q_METHODS = ['ADA', 'GPTQ', 'TORCH_INT', 'BITSANDBYTES']

def is_available_bit(bit):
    assert bit in AVAILABLE_BITS, f"bit {bit} is not available, please choose from {AVAILABLE_BITS}"

def is_available_q_method(method):
    assert method in AVAILABLE_Q_METHODS, f"Q_METHOD {method} is not available, please choose from {AVAILABLE_Q_METHODS}"

import json
import os 
EXTRA_METHODS = []
with open(os.path.join(os.path.dirname(__file__), "extra_methods.json")) as f:
    data = json.load(f)
    EXTRA_METHODS = data['extra_q_methods']
    
