import torch
from torch import nn
from model.edsr_bireal_fortest import make_model

if __name__ == '__main__':
    from option import args
    args.binary_mode = 'binary'
    args.n_resblocks = 16
    args.n_feats = 64
    args.scale = [4]
    args.n_colors = 1
    args.rgb_range = 256
    args.res_scale = 1

    x = torch.randn(1,1,256,256)
    model = make_model(args)
    y = model(x)
    print(y.shape)