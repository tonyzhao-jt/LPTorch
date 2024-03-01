from setuptools import setup, find_packages
from torch.utils import cpp_extension

from pathlib import Path
import os

import subprocess

from typing import List

ROOT_DIR = Path(__file__).parent

def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)

# requirements
def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements

# check if it is the GPU with cap < 80
def bitsandbytes_below_80():
    GPU_name = os.popen('(CUDA_VISIBLE_DEVICES=0 nvidia-smi --query-gpu=name --format=csv,noheader | tr " " _)').readlines()[0].strip()
    GPU_name_upper = GPU_name.upper()
    if 'A100' not in GPU_name_upper and 'H100' not in GPU_name_upper:
        install_commands = [
            "cd 3rd_party && bash update_3rd.sh",
        ]
        for command in install_commands:
            subprocess.call(command, shell=True)

setup(
    name='lptorch',
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='lptorch.gptq._CUDA',
            sources=[
                'lptorch/gptq/kernels/quant_cuda_kernel.cu',
                'lptorch/gptq/kernels/bindings.cpp',
            ],
            include_dirs=['lptorch/gptq/kernels/include'],
            extra_link_args=['-lcublas', '-lcublasLt', '-lcudart', 
                             '-lrt', '-lpthread', '-ldl', '-L/usr/local/cuda/lib64/'],
        ),
        cpp_extension.CUDAExtension(
            name='lptorch.torch_int._CUDA',
            sources=[
                'lptorch/torch_int/kernels/linear.cu',
                'lptorch/torch_int/kernels/bmm.cu',
                'lptorch/torch_int/kernels/fused.cu',
                'lptorch/torch_int/kernels/bindings.cpp',
            ],
            include_dirs=[
                'lptorch/torch_int/kernels/include',
                'lptorch/cutlass/include',
                'lptorch/cutlass/util/include',
            ],
            extra_link_args=['-lcublas', '-lcublasLt', '-lcudart', 
                             '-lrt', '-lpthread', '-ldl', '-L/usr/local/cuda/lib64/'],
            extra_compile_args={'cxx': ['-std=c++14', '-O3'],
                                'nvcc': ['-O3', '-std=c++14', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__']},
        ),
        cpp_extension.CUDAExtension(
            name='lptorch._CUDA',
            sources=[
                'lptorch/lptorch/kernels/linear.cu',
                'lptorch/lptorch/kernels/bmm.cu',
                'lptorch/lptorch/kernels/fused.cu',
                'lptorch/lptorch/kernels/bindings.cpp',
            ],
            include_dirs=[
                'lptorch/lptorch/kernels/include',
                'lptorch/cutlass/include',
                'lptorch/cutlass/util/include',
            ],
            extra_link_args=['-lcublas', '-lcublasLt', '-lcudart', 
                             '-lrt', '-lpthread', '-ldl', '-L/usr/local/cuda/lib64/'],
            extra_compile_args={'cxx': ['-std=c++14', '-O3'],
                                'nvcc': ['-O3', '-std=c++14', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__']},
        ),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False)
    },
    packages=find_packages(
        exclude=['notebook', 'scripts', 'tests', 'bench']),
    package_data={
        'lptorch': ['utils/device_info.csv', 'config/extra_methods.json'],
    },
    install_requires=get_requirements(),
)