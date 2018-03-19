import torch
from torch import nn, cuda, autograd
from model.PerceptualLoss import PerceptualLoss
from model.ArtisticNet import ArtisticNet
import argparse
import torchvision.models as models
import torchvision.transforms as transforms
# TODO 实现数据集读取

parser = argparse.ArgumentParser(description="视频风格迁移训练参数配置")

parser.add_argument("--use_instance_norm", default=1)
parser.add_argument("--padding_type", default='reflect')
parser.add_argument("--tanh_constant", default=150)
parser.add_argument("--tv_strength", default=1e-6)
parser.add_argument("--content_weights", default=1.0)
parser.add_argument("--style_weights", default=5.0)

# Optimization
parser.add_argument("--num_iterations", default=40000)
parser.add_argument("--batch_size", default=4)
parser.add_argument("--learning_rate", default=1e-3)
args = parser.parse_args()

def main():
    net = ArtisticNet(args)
    cnn = models.vgg19(pretrained=True).features
    if cuda.is_available():
        cnn = cnn.cuda()