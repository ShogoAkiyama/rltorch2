import numpy as np

def standard_norm(data):
    means = data.mean(axis =1)
    stds = data.std(axis= 1, ddof=1)
    data = (data - means[:, np.newaxis]) / stds[:, np.newaxis]
    return np.nan_to_num(data)

