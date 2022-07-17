import functools
from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return RCAN(args)

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

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, conv=nn.Conv2d):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                conv(channel, channel // reduction, 1, padding=0, bias=True),
                # nn.ReLU(inplace=True),
                nn.LeakyReLU(inplace=True),
                conv(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

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

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        # modules_body = []
        # for i in range(2):
        #     modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        #     if bn: modules_body.append(nn.BatchNorm2d(n_feat))
        #     if i == 0: modules_body.append(act)
        # modules_body.append(CALayer(n_feat, reduction))
        # self.body = nn.Sequential(*modules_body)
        self.conv1 = nn.Sequential(
            conv(n_feat, n_feat, kernel_size, padding=kernel_size//2, bias=bias),
            nn.BatchNorm2d(n_feat)
        )
        self.act = act()
        self.conv2 = nn.Sequential(
            conv(n_feat, n_feat, kernel_size, padding=kernel_size//2, bias=bias),
            nn.BatchNorm2d(n_feat)
        )
        self.ca = CALayer(n_feat, reduction)
        self.res_scale = res_scale

    def forward(self, x):
        # res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        # res += x
        out = self.conv1(x) + x
        out = self.act(out)
        out = self.conv2(x) + out
        out = self.ca(out)
        return out

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=res_scale) \
            for _ in range(n_resblocks)]
        # modules_body.append(conv(n_feat, n_feat, kernel_size, padding=1))
        modules_body.append(BasicBlock(conv, n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, args):
        super(RCAN, self).__init__()
        
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        reduction = args.reduction
        kernel_size = 3
        scale = args.scale[0]
        act = functools.partial(nn.LeakyReLU, negative_slope=0.1, inplace=True)
        
        conv = functools.partial(BinaryConv, mode=args.binary_mode)
        
        # RGB mean for DIV2K
        if args.n_colors == 3:
            self.sub_mean = common.MeanShift(args.rgb_range)
            self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        else:
            self.sub_mean = common.MeanShift(args.rgb_range, n_colors=1, rgb_mean=[0.5], rgb_std=[1])
            self.add_mean = common.MeanShift(args.rgb_range, n_colors=1, rgb_mean=[0.5], rgb_std=[1], sign=1)
        
        # define head module
        modules_head = [nn.Conv2d(args.n_colors, n_feats, kernel_size, padding=1)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        # modules_body.append(conv(n_feats, n_feats, kernel_size, padding=1))

        # define tail module
        # modules_tail = [
        #     common.Upsampler(conv, scale, n_feats, act=False),
        #     nn.Conv2d(n_feats, args.n_colors, kernel_size, padding=1)]
        modules_tail = [
            nn.Conv2d(n_feats, args.n_colors * scale**2, 3, padding=1),
            nn.PixelShuffle(scale)
        ]

        # self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))