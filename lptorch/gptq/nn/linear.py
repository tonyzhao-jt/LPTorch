import numpy as np
import torch
import torch.nn as nn
import math

try:
    import lptorch.gptq.functional as qfunc
except:
    print('CUDA extension not installed.')

# Assumes layer is perfectly divisible into 256 * 256 blocks
class QuantLinear(nn.Module): 
    def __init__(self, bits, groupsize, infeatures, outfeatures):
        super().__init__()
        if bits not in [2,3,4,8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        if groupsize != -1 and groupsize < 32 and groupsize != int(math.pow(2,int(math.log2(groupsize)))):
            raise NotImplementedError("groupsize supports powers of 2 greater than 32. (e.g. : 32,64,128,etc)")
        groupsize = groupsize if groupsize != -1 else infeatures
        self.groupsize = groupsize
        self.register_buffer('qzeros', torch.zeros((math.ceil(infeatures/groupsize),outfeatures // 256 * (bits * 8)), dtype=torch.int))
        self.register_buffer('scales', torch.zeros((math.ceil(infeatures/groupsize),outfeatures)))
        self.register_buffer('bias', torch.zeros(outfeatures))
        self.register_buffer(
            'qweight', torch.zeros((infeatures // 32 * bits, outfeatures), dtype=torch.int)
        )
        self._initialized_quant_state = False

    def pack(self, linear, scales, zeros):
        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone()
        if linear.bias is not None:
            self.bias = linear.bias.clone() 
            
        intweight = []
        for idx in range(self.infeatures):
            g_idx = idx // self.groupsize
            intweight.append(torch.round((linear.weight.data[:,idx] + scale_zeros[g_idx]) / self.scales[g_idx]).to(torch.int)[:,None])
        intweight = torch.cat(intweight,dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros(
            (intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32
        )
        i = 0
        row = 0
        while row < qweight.shape[0]:
            if self.bits in [2,4,8]:
                for j in range(i, i + (32//self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32//self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")
                
        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight) 
        
        zeros -= 1
        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 256 * (self.bits * 8)), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2,4,8]:
                for j in range(i, i + (32//self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32//self.bits
                col += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i))
                i += 10
                qzeros[:, col] |= zeros[:, i] << 30
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 1)
                i += 10
                qzeros[:, col] |= zeros[:, i] << 31
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 2)
                i += 10
                col += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")
                
        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros) 

    def forward(self, x):
        intermediate_dtype = torch.float32

        if not self._initialized_quant_state:
            # Do we even have a bias? Check for at least one non-zero element.
            if self.bias is not None and bool(torch.any(self.bias != 0)):
                # Then make sure it's the right type.
                self.bias.data = self.bias.data.to(intermediate_dtype)
            else:
                self.bias = None

        outshape = list(x.shape)
        outshape[-1] = self.outfeatures
        x = x.reshape(-1, x.shape[-1])
        if self.bias is None:
            y = torch.zeros(x.shape[0], outshape[-1], dtype=intermediate_dtype, device=x.device)
        else:
            y = self.bias.clone().repeat(x.shape[0], 1)

        output_dtype = x.dtype
        x = x.to(intermediate_dtype)
        if self.bits == 2:
            qfunc.vecquant2matmul_torch(x, self.qweight, y, self.scales, self.qzeros, self.groupsize)
        elif self.bits == 3:
            qfunc.vecquant3matmul_torch(x, self.qweight, y, self.scales, self.qzeros, self.groupsize)
        elif self.bits == 4:
            qfunc.vecquant4matmul_torch(x, self.qweight, y, self.scales, self.qzeros, self.groupsize)
        elif self.bits == 8:
            qfunc.vecquant8matmul_torch(x, self.qweight, y, self.scales, self.qzeros, self.groupsize)
        else:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        y = y.to(output_dtype)
        return y.reshape(outshape)
