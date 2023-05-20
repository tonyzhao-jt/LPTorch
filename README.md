# LPTorch
A Torch Plugin That Support Low-Precision Kernel Implementation


# NOTICE
MUST RUN ON CUDA, MUST RUN ON CUDA, MUST RUN ON CUDA
- you can initilzie the AdaQLinear in CPU but must run test on CUDA.


# FasterTransformer
build from source.
```
mkdir build
cd build
cmake -DSM=80 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/mpi/lib:/opt/conda/lib/python3.10/site-packages/torch/lib:/opt/conda/lib
make -j12
```
# load lib
self.ft_longformer_lib = os.path.join('build', 'lib', 'libth_transformer.so')
torch.classes.load_library(ft_longformer_lib)

# issues
https://github.com/NVIDIA/FasterTransformer/issues/69
- add CMAKELISTS: -lmpi_cxx -lmpi
https://github.com/NVIDIA/FasterTransformer/issues/261
- add src/fastertransformer/utils/CMakeLists.txt
- `-lmpi_cxx` for 'mpi_utils'