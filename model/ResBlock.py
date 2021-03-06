from torch import nn
import torch
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, dim, opt):
        super(ResBlock, self).__init__()
        self.opt = opt
        self.conv_block = self.build_conv_block(dim, opt.padding_type, opt.use_instance_norm)

    def build_conv_block(self, dim, padding_type, use_instance_norm=True):
        conv_block = nn.Sequential()
        p = 0
        if padding_type == 'reflect':
            conv_block.add_module('relect_padding1', nn.ReflectionPad2d((1, 1, 1, 1)))
        elif padding_type == 'replicate':
            conv_block.add_module("replicate_padding1", nn.ReplicationPad2d((1, 1, 1, 1)))
        elif padding_type == 'zero':
            p = 1
        conv_block.add_module('conv_layer1', nn.Conv2d(dim, dim, 3, 1, p))
        if use_instance_norm:
            conv_block.add_module('IN', nn.InstanceNorm2d(dim))
        else:
            conv_block.add_module('BN', nn.BatchNorm2d(dim))
        conv_block.add_module('relu', nn.ReLU(True))
        if padding_type == 'reflect':
            conv_block.add_module('relect_padding2', nn.ReflectionPad2d((1, 1, 1, 1)))
        elif padding_type == 'replicate':
            conv_block.add_module("replicate_padding2", nn.ReplicationPad2d((1, 1, 1, 1)))
        conv_block.add_module('conv_layer2', nn.Conv2d(dim, dim, 3, 1, p))
        if use_instance_norm:
            conv_block.add_module('IN', nn.InstanceNorm2d(dim))
        else:
            conv_block.add_module('BN', nn.BatchNorm2d(dim))
        return conv_block

    def forward(self, x):
        output = self.conv_block(x)
        assert output.shape == x.shape
        return torch.add(output, x)
