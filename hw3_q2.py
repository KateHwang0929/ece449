import hw3_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


def svm_solver(x_train, y_train, lr, num_iters,
               kernel=hw3_utils.poly(degree=1), c=None):
    n = x_train.shape[0]
    y = y_train.to(device=x_train.device, dtype=x_train.dtype)
    K = torch.stack([
        torch.stack([
            torch.as_tensor(kernel(x_train[i], x_train[j]),
                            dtype=x_train.dtype, device=x_train.device)
            for j in range(n)
        ])
        for i in range(n)
    ])
    Q = (y[:, None] * y[None, :]) * K
    alpha = torch.zeros(n, dtype=x_train.dtype, device=x_train.device)

    for _ in range(num_iters):
        gradient = Q @ alpha - 1
        alpha = alpha - lr * gradient
        if c is None:
            alpha.clamp_(min=0)
        else:
            alpha.clamp_(min=0, max=c)

    return alpha.detach()


def svm_predictor(alpha, x_train, y_train, x_test, kernel=hw3_utils.poly(degree=1)):
    alpha = alpha.to(device=x_train.device, dtype=x_train.dtype)
    y = y_train.to(device=x_train.device, dtype=x_train.dtype)
    weighted = alpha * y
    support_indices = torch.nonzero(alpha > 0, as_tuple=False).flatten()
    if support_indices.numel() == 0:
        raise ValueError('No support vectors found: all alpha values are zero.')
    support_index = support_indices[torch.argmin(alpha[support_indices])].item()

    def decision_without_bias(point):
        kernel_values = torch.stack([
            torch.as_tensor(kernel(x_train[j], point), dtype=x_train.dtype, device=x_train.device)
            for j in range(x_train.shape[0])])
        return torch.dot(weighted, kernel_values)

    b = y[support_index] - decision_without_bias(x_train[support_index])
    predictions = [decision_without_bias(point) + b for point in x_test]
    return torch.stack(predictions)
