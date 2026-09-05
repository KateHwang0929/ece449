import numpy
import torch

def numpy_squares(k):
    x = numpy.arange(1, k + 1)
    return x ** 2

def torch_squares(k):
    x = torch.arange(1, k + 1)
    return x ** 2