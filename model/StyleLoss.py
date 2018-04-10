from torch import nn
from model import GramMatrix
import torch


class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach()
        self.weight = weight
        self.gram = GramMatrix.GramMatrix()
        self.criterion = nn.MSELoss().cuda()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        N = input.size(2) * input.size(3)
        M = input.size(1)
        if self.G.size() == self.target.repeat(input.size(0), 1, 1).size():
            loss = torch.sum(torch.pow(self.G - self.target.repeat(input.size(0), 1, 1), 2))
            self.loss = loss * self.weight / (2 * input.size(0) * N ** 2 * M ** 2)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss
