import torch.nn as nn
import torch


class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight

    def forward(self, input):
        self.output = input.clone()
        if input.size() == self.target.size():
            N = input.size(2) * input.size(3)
            M = input.size(1)
            loss = torch.sum(torch.pow(input-self.target,2))
            self.loss = loss*self.weight/(2*input.size(0))
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss
