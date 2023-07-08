# refer to: https://github.com/TimDettmers/bitsandbytes/issues/165
# bitsandbytes need to be altered for cap <=70 
cp _functions.py ./bitsandbytes/bitsandbytes/autograd/_functions.py
cd bitsandbytes
CUDA_VERSION=117 make cuda11x_nomatmul # check whether your device is 117
# python setup.py install # won't detect the new .so as well
pip install .