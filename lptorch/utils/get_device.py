
import sys
import os
import pandas as pd
DEVICE_INFO = pd.read_csv(os.path.join(os.path.dirname(__file__), "device_info.csv"))
DEVICE_INFO.set_index("device_name", inplace = True)

ALIAS_DICT = {
    "A100": ["A100-SXM4-40GB", "NVIDIA_A100-SXM4-40GB", "A100_Graphics_Device", "A100-SXM-80GB"],
    "V100": ["Tesla_V100-SXM2-32GB", "Tesla_V100-PCIE-32GB", "Tesla_V100-SXM2-16GB"],
    "T4": ["Tesla_T4"],
    "A100-SXM4-40GB_1g_5gb": [],
    "A100-SXM4-40GB_2g_10gb": [],
    "A100-SXM4-40GB_3g_20gb": [],
    "A100-SXM4-40GB_4g_20gb": [],
    "A30": [],
    "A10": ["NVIDIA_A10"],
    "K80": [],
    "P100": ["Tesla_P100-PCIE-12GB"],
    "HL-100": []
}

ALIAS2SHORT = {}

for gpu_model, alias in ALIAS_DICT.items():
    ALIAS2SHORT[gpu_model] = gpu_model
    for alia in alias:
        ALIAS2SHORT[alia.upper()] = gpu_model

### **NOTE**: ALL_GPU_MODEL must be modified manually, instead of being generated from ALIAS2SHORT
# Be careful to change the order of gpu names, because changing the order of a gpu model 
# name may affect the current cached profiled data
ALL_GPU_MODEL = [
    "A100-SXM4-40GB_1g_5gb",
    "A100-SXM4-40GB_2g_10gb",
    "A100-SXM4-40GB_3g_20gb",
    "A100_Graphics_Device",
    "A100-SXM4-40GB_4g_20gb",
    "A100-SXM4-40GB",
    "Tesla_T4",
    "A30",
    "Tesla_V100-SXM2-32GB",
    "A100",
    "T4",
    "V100",
    "A10",
    "K80",
    "P100",
    "HL-100"
]

def gpu_model2int(gpu_model):
    return ALL_GPU_MODEL.index(gpu_model)

def short_device_name(gpu_model):
    return ALIAS2SHORT[gpu_model]

def query_cc(gpu_model):
    ''' Query the compute capability '''
    return int(DEVICE_INFO.loc[short_device_name(gpu_model.upper())].capability)

def query_core_num(gpu_model, dtype, tensor_core_or_not):
    ''' Return the number of arithmetic units corresponding to `dtype` '''
    if dtype == "fp16":
        if tensor_core_or_not:
            return DEVICE_INFO[short_device_name(gpu_model)]["tensor_core_au_per_sm"]
        elif "fp16_au_per_sm" in DEVICE_INFO[short_device_name(gpu_model)]:
            return DEVICE_INFO[short_device_name(gpu_model)]["fp16_au_per_sm"]
        else:
            return DEVICE_INFO[short_device_name(gpu_model)]["fp32_au_per_sm"]
    elif dtype == "fp32":
        return DEVICE_INFO[short_device_name(gpu_model)]["fp32_au_per_sm"]


def get_capability():
    device = os.popen('(CUDA_VISIBLE_DEVICES=0 nvidia-smi --query-gpu=name --format=csv,noheader | tr " " _)').readlines()
    device_0 = device[0].strip()
    capability = query_cc(device_0)
    return capability

def is_tensorcore_int8_available():
    capability = get_capability()
    return capability >= 75 # only for A100,T4,H100

def is_tensorcore_int8_available_offline(device_0):
    capability = query_cc(device_0)
    return capability >= 75 # only for A100,T4,H100