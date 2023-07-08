# LPTorch
A Torch Plugin That Support Low-Precision Kernel Implementation


# NOTICE
MUST RUN ON CUDA, MUST RUN ON CUDA, MUST RUN ON CUDA
- you can initilzie the AdaQLinear in CPU but must run test on CUDA.

For Capability < 70, go `3rd_party` and run 
```bash
    bash update_3rd.sh
```
to recompile the bitsandbytes, modify the CUDA if necessary. 

# Support
|Name|Precisions|Link| Desc |
|---|---|---|---|
|SPQR-3bit| 3,4 |https://github.com/Vahe1994/SpQR||
|AWQ|4|https://github.com/mit-han-lab/llm-awq||
|SmoothQuant| 8 | https://github.com/mit-han-lab/smoothquant|We further add the support for >=75|
|LLM.int8()| 8 | https://github.com/TimDettmers/bitsandbytes/tree/main/bitsandbytes| HuggingFace, No Training Required |