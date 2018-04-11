from torch import nn
from model import GramMatrix
import torch


class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach()
        self.weight = weight
        self.gram = GramMatrix.GramMatrix()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        if self.G.size() == self.target.repeat(input.size(0), 1, 1).size():
            loss = torch.dist(self.G, self.target.repeat(input.size(0), 1, 1), 2)
            self.loss = loss * self.weight / (4 * input.size(0))
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss
