import numpy as np
import torch

def index2onehot(index, dim):
    if isinstance(index, np.int) or isinstance(index, np.int64):
        one_hot = np.zeros(dim)
        one_hot[index] = 1.
    else:  # indexが多次元
        one_hot = np.zeros((len(index), dim))
        one_hot[np.arange(len(index)), index] = 1.
    return one_hot

def entropy(p):
    return -torch.sum(p * torch.log(p), 1)
