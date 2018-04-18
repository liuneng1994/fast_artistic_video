import torch
from torch import autograd


class TVLoss2(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = torch.zeros(input.size())
        x_diff = input.clone()[:, :, 1:-2, 1:-2]
        x_diff = x_diff - input[:, :, 1:-2, 2, -1]
        y_diff = input.clone()[:, :, 1:-2, 1:-2]
        y_diff = y_diff - input[:, :, 2:-1, 1:-2]
        grad_input[:, :, 1:-2, 1:-2].add(x_diff).add(y_diff)
        grad_input[:, :, 1:-2, 2:-1].add(-1, x_diff)
        grad_input[:, :, 2:-1, 1:-2].add(-1, y_diff)
        grad_input = grad_input * 1e-6
        grad_input += grad_output
        return grad_input
