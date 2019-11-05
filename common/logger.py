import numpy as np
from collections import deque


class RewardLogger:

    def __init__(self, beta=0.2):
        # reward deque
        self._deque = deque(maxlen=100)
        # smoothing parameter
        self._beta = beta

    def mean_reward(self):
        # avoid arena with not enough samples
        if len(self._deque) <= 50:
            return 0.0
        else:
            return np.array(self._deque).mean()

    def add_reward(self, reward):
        assert isinstance(reward, float) or isinstance(reward, int)

        # append
        self._deque.append(reward)
