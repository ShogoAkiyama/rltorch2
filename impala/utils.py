import numpy as np
import torch


def idx2onehot(idx, batch, dim):
    if isinstance(idx, np.int) or isinstance(idx, np.int64):
        one_hot = np.zeros(dim)
        one_hot[idx] = 1.
    else:  # indexが多次元
        one_hot = torch.eye(batch, dim)[idx]
    return one_hot

def entropy(p):
    return -torch.sum(p * torch.log(p), 1)
