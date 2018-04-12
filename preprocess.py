import torch
import numpy as np

mean = torch.FloatTensor(np.asarray([103.939, 116.779, 123.68]).reshape((3, 1, 1)))


def preprocess(img):
    return torch.mul(img, 255) - mean


def deprocess(img):
    return (img + mean) / 255
