import torch 
from lptorch.gptq import _CUDA
# import vecquant2matmul, vecquant3matmul, vecquant4matmul, vecquant8matmul

# pack them with function
def vecquant2matmul_torch(vec, mat, mul, scales, zeros, M):
    return _CUDA.vecquant2matmul(vec, mat, mul, scales, zeros, M)

def vecquant3matmul_torch(vec, mat, mul, scales, zeros, M):
    return _CUDA.vecquant3matmul(vec, mat, mul, scales, zeros, M)

def vecquant4matmul_torch(vec, mat, mul, scales, zeros, M):
    return _CUDA.vecquant4matmul(vec, mat, mul, scales, zeros, M)

def vecquant8matmul_torch(vec, mat, mul, scales, zeros, M):
    return _CUDA.vecquant8matmul(vec, mat, mul, scales, zeros, M)