from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

def make_model(args, parent=False):
    return EDSR(args)


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

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, act=functools.partial(nn.LeakyReLU, negative_slope=0.1, inplace=True), res_scale=1):

        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv(n_feats, n_feats, kernel_size, padding=kernel_size//2, bias=bias),
            nn.BatchNorm2d(n_feats)
        )
        self.act = act()
        self.conv2 = nn.Sequential(
            conv(n_feats, n_feats, kernel_size, padding=kernel_size//2, bias=bias),
            nn.BatchNorm2d(n_feats)
        )
        self.res_scale = res_scale

    def forward(self, x):
        # res = self.body(x).mul(self.res_scale)
        # res += x
        out = self.conv1(x).mul(self.res_scale) + x
        out = self.act(out)
        out = self.conv2(out).mul(self.res_scale) + out
        return out


class UpSamper(nn.Module):
    def __init__(self, conv, channels, scale, n_colors):
        super(UpSamper, self).__init__()
        self.scale = scale
        self.conv1 = conv(channels, channels*scale**2, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels*scale**2)
        self.pixelshuffle = nn.PixelShuffle(scale)
        self.conv2 = nn.Conv2d(channels, n_colors, 3, padding=1)

    def forward(self, x):
        fea = self.bn1(self.conv1(x)) + torch.repeat_interleave(x, self.scale**2, dim=1)
        fea = self.pixelshuffle(fea)
        fea = self.conv2(fea)
        return fea


class EDSR(nn.Module):
    def __init__(self, args, conv=BinaryConv):
        super(EDSR, self).__init__()

        conv = functools.partial(conv, mode=args.binary_mode)

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        print(args.n_feats)
        kernel_size = 3 
        scale = args.scale[0]
        act = functools.partial(nn.LeakyReLU, negative_slope=0.1, inplace=True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        if args.n_colors == 3:
            self.sub_mean = common.MeanShift(args.rgb_range)
            self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        else:
            self.sub_mean = common.MeanShift(args.rgb_range, n_colors=1, rgb_mean=[0.5], rgb_std=[1])
            self.add_mean = common.MeanShift(args.rgb_range, n_colors=1, rgb_mean=[0.5], rgb_std=[1], sign=1)

        # define head module
        m_head = [BasicBlock(nn.Conv2d, args.n_colors, n_feats, kernel_size, bn=False)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        # m_body.append(nn.Sequential(
        #     conv(n_feats, n_feats, kernel_size, padding=kernel_size//2),
        #     nn.BatchNorm2d(n_feats)
        # ))

        # define tail module
        # m_tail = [
        #     common.Upsampler(conv, scale, n_feats, bn=False, act=False),
        #     BasicBlock(nn.Conv2d, n_feats, args.n_colors, kernel_size, bn=False)
        # ]
        m_tail = [
            UpSamper(conv, n_feats, scale, args.n_colors)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

