import numpy as np
from torch import nn


class QRDQN(nn.Module):

    def __init__(self, num_states, num_actions, N=200,
                 sensitive=None, c=0):
        super(QRDQN, self).__init__()

        # Quantile network.
        self.q_net = nn.Sequential(
            nn.Linear(num_states, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions * N),
        )

        self.N = N
        self.num_actions = num_actions
        self.sensitive = sensitive
        self.c = c
        self.num_cvar = int(np.ceil(self.N * self.c))

    def forward(self, states=None):
        batch_size = states.shape[0]

        quantiles = self.q_net(states).view(
            batch_size, self.N, self.num_actions)

        assert quantiles.shape == (batch_size, self.N, self.num_actions)

        return quantiles

    def calculate_q(self, states=None):
        batch_size = states.shape[0]

        # Calculate quantiles.
        quantiles = self(states=states)

        if self.c == 0:
            quantiles = quantiles
        elif self.sensitive:
            quantiles = quantiles[:, :self.num_cvar]
        else:
            quantiles = quantiles[:, -self.num_cvar+1:]

        # Calculate expectations of value distributions.
        q = quantiles.mean(dim=1)
        assert q.shape == (batch_size, self.num_actions)

        return q
