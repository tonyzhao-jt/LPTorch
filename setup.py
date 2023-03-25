from setuptools import setup, find_packages
from torch.utils import cpp_extension

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
            extra_link_args=['-lcublas_static', '-lcublasLt_static',
                             '-lculibos', '-lcudart', '-lcudart_static',
                             '-lrt', '-lpthread', '-ldl', '-L/usr/lib/x86_64-linux-gnu/'],
        ),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False)
    },
    packages=find_packages(
        exclude=['notebook', 'scripts', 'tests']),
)