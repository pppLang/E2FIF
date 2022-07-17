# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

import functools
from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_model(args, parent=False):
    return RDN(args)


class Q_A(torch.autograd.Function):  # dorefanet, but constrain to {-1, 1}
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()                     
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = (2 - torch.abs(2*input))
        grad = grad_input.clamp(0) * grad_output.clone()
        return grad

class Q_W(torch.autograd.Function):  # xnor-net, but gradient use identity approximation
    @staticmethod
    def forward(ctx, x):
        return x.sign()
    @staticmethod
    def backward(ctx, grad):
        return grad

class BinaryConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, bitW=1, stride=1, padding=0, bias=True, groups=1, mode='binary'):
        super(BinaryConv, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
        self.groups = groups
        self.bitW = bitW
        self.padding = padding
        self.stride = stride
        self.change_nums = 0
        self.mode = mode
        assert self.mode in ['pretrain', 'binary', 'binaryactonly']
        print('conv mode : {}'.format(self.mode))

    def forward(self, input):

        if self.mode == 'binaryactonly' or self.mode == 'binary':
            input = Q_A.apply(input)
        elif self.mode == 'pretrain':
            pass
        else:
            assert False
            
        if self.mode == 'binaryactonly' or self.mode == 'pretrain':
            weight = self.weight
        elif self.mode == 'binary':
            weight = Q_W.apply(self.weight)
        else:
            assert False
        output = F.conv2d(input, weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return output

class BasicBlock(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, bn=True):

        super(BasicBlock, self).__init__()
        self.residual = in_channels == out_channels
        m = [conv(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False if bn else True)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        self.conv1 = nn.Sequential(*m)

    def forward(self, x):
        if self.residual:
            return self.conv1(x) + x
        else:
            return self.conv1(x)


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3, conv=None):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            BasicBlock(conv, Cin, G, kSize),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1), out

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3, conv=None):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G, conv=conv))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = BasicBlock(conv, G0 + C*G, G0, 1)

    def forward(self, x):
        # return self.LFF(self.convs(x)) + x
        x0 = x
        for conv in self.convs:
            x, x_this = conv(x)
        x = self.LFF(x) + x_this
        return x + x0

class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        r = args.scale[0]
        G0,self.D, C, G = args.n_feats, args.n_resblocks, args.n_convs, args.n_feats
        kSize = args.RDNkSize
        print(G0,self.D, C, G)

        conv = functools.partial(BinaryConv, mode=args.binary_mode)

        # number of RDB blocks, conv layers, out channels
        # self.D, C, G = {
        #     'A': (20, 6, 32),
        #     'B': (16, 8, 64),
        # }[args.RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = BasicBlock(conv, G0, G0, kSize)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C, conv=conv)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            BasicBlock(conv, self.D * G0, G0, 1),
            BasicBlock(conv, G0, G0, kSize),
        ])

        # Up-sampling net
        self.UPNet = nn.Sequential(*[
            nn.Conv2d(G, args.n_colors*r**2, kSize, padding=(kSize-1)//2, stride=1),
            nn.PixelShuffle(r),
        ])

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1)) + x
        x += f__1

        return self.UPNet(x)