import torch
from torch import autograd


class ShaveImage(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        H, W = input.size(2), input.size(3)
        s = 2
        ctx.save_for_backward(input, s)
        output = input[:, :, s + 1:H - s, s + 1:W - s]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, s = ctx.saved_tensors
        H, W = input.size(2), input.size(3)
        grad_input = grad_output.clone()
        grad_input = grad_input[:, :, s + 1:H - s, s + 1:W - s]
        return grad_input
