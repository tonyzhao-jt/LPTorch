import os
import torch
# get current path
current_path = os.path.dirname(os.path.realpath(__file__))
ft_longformer_lib = os.path.join(current_path, '../3rd_party/FasterTransformer', 'build', 'lib', 'libth_transformer.so')
torch.classes.load_library(ft_longformer_lib)