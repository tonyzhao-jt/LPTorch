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


# PS
This repo is built to power the project LLM-PQ. Please Cite the paper if you find the repo is useful to you.

our [paper](https://dl.acm.org/doi/10.1145/3627535.3638480):
```bibtex
@inproceedings{10.1145/3627535.3638480,
author = {Zhao, Juntao and Wan, Borui and Wu, Chuan and Peng, Yanghua and Lin, Haibin},
title = {POSTER: LLM-PQ:Serving LLM on Heterogeneous Clusters with Phase-Aware Partition and Adaptive Quantization},
year = {2024},
isbn = {9798400704352},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3627535.3638480},
doi = {10.1145/3627535.3638480},
pages = {460–462},
keywords = {LM serving, heterogenous cluster, quantization},
series = {PPoPP '24}
}
```