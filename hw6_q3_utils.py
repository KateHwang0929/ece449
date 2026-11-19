import random

import numpy as np
import torch

from hw6_q3 import hw6_q3_autograd as ad
from hw6_q3 import hw6_q3_nn as nn

torch.use_deterministic_algorithms(True)
torch.set_default_dtype(torch.float64)

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_num_params(parameters):
    if isinstance(parameters, list) and isinstance(parameters[0], ad.Scalar):
        return len(parameters)
    return sum(param.numel() for param in parameters)

def compute_batch(mlp, X):
    if isinstance(mlp, nn.Module):
        if isinstance(X, np.ndarray):
            X = [[ad.Scalar(float(xi)) for xi in x] for x in X]
        return [mlp(Xi) for Xi in X] 
    else:
        return mlp(X).squeeze(1) 

def accuracy(mlp, X, y):
    logits = compute_batch(mlp, X)

    def is_correct(logit, label):
        return (logit.item() >= 0) == (label == 1)

    if isinstance(y, np.ndarray):
        y = y.tolist()
    
    correct = sum(is_correct(logit, label) for logit, label in zip(logits, y))
    return correct/len(logits)


def train(mlp, ss_fn, optimizer, X, y, batch_size=8, num_steps=100):

    set_seed()
    losses, accs = [], []
    for i in range(1, num_steps + 1):
        batch_ids = random.sample(range(len(X)), k=batch_size)
        batch_X = X[batch_ids]
        batch_y = y[batch_ids]

        ### YOUR IMPLEMENTATION START ###
        outputs = compute_batch(mlp, batch_X)
        loss = loss_fn(outputs, batch_y)
        mlp.zero_grad()
        loss.backward()
        optimizer.step()
        ### YOUR IMPLEMENTATION END ###

        acc = accuracy(mlp, X, y)
        print(f"Step {i:#3d} \t Loss {loss.item():#.4f} \t Accuracy {acc}")

        losses.append(loss.item())
        accs.append(acc)

    return losses, accs
