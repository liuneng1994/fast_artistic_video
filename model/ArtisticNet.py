import torch
from torch import nn, autograd,cuda
import torch.nn.functional as F
from model.ResBlock import ResBlock
from model.TVLoss import TVLoss


class ArtisticNet(nn.Module):
    def forward(self, x):
        output = self.model(x)
        return torch.mul(output, self.opt['tanh_constant'])

    def __init__(self, opt):
        super(ArtisticNet, self).__init__()
        self.opt = opt

    def conv(self, filter_size, stride, in_dim, out_dim):
        p = (filter_size - 1) / 2
        p = int(p)
        conv_model = nn.Sequential()
        if self.opt['padding_type'] == 'replicate':
            conv_model.add_module("pad", nn.ReplicationPad2d(p))
            p = 0
        elif self.opt['padding_type'] == 'reflect':
            conv_model.add_module('pad', nn.ReflectionPad2d(p))
            p = 0
        elif self.opt['padding_type'] == 'none':
            pass
        conv_model.add_module('conv', nn.Conv2d(in_dim, out_dim, filter_size, stride, padding=p))
        return conv_model

    @staticmethod
    def down_sample(in_dim, out_dim):
        return nn.Conv2d(in_dim, out_dim, 3, 2, 1)

    @staticmethod
    def up_sample(in_dim, out_dim):
        return nn.ConvTranspose2d(in_dim, out_dim, 3, 2, 1, 1)

    def build_model(self):
        model = nn.Sequential()
        model.add_module('conv_1', self.conv(9, 1, 3, 32))
        model.add_module('down1', self.down_sample(32, 64))
        model.add_module('down2', self.down_sample(64, 128))
        model.add_module('res1', ResBlock(128, self.opt))
        model.add_module('res2', ResBlock(128, self.opt))
        model.add_module('res3', ResBlock(128, self.opt))
        model.add_module('res4', ResBlock(128, self.opt))
        model.add_module('res5', ResBlock(128, self.opt))
        model.add_module('up1', self.up_sample(128, 64))
        model.add_module('up2', self.up_sample(64, 32))
        model.add_module('conv_2', self.conv(9, 1, 32, 3))
        model.add_module('tanh', nn.Tanh())
        if cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model


if __name__ == '__main__':
    model = ArtisticNet({'use_instance_norm': True})
    model.build_model()
    a = model(autograd.Variable(torch.randn(4, 3, 256, 256)))
    model('res1')
