import torch
from torch import nn
import numpy as np


class ConsistencyChecker:
    def checkConsistency(flow1, flow2, reliable):
        x_size = flow1.shape(0)
        y_size = flow1.shape(1)
        dx = torch.FloatTensor(x_size, y_size, 2)
        dy = torch.FloatTensor(x_size, y_size, 2)

