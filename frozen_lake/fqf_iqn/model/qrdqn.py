from torch import nn


class QRDQN(nn.Module):

    def __init__(self, num_states, num_actions, N=200):
        super(QRDQN, self).__init__()

        # Quantile network.
        self.q_net = nn.Sequential(
            nn.Linear(num_states, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions * N),
        )

        self.N = N
        self.num_actions = num_actions

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

        # Calculate expectations of value distributions.
        q = quantiles.mean(dim=1)
        assert q.shape == (batch_size, self.num_actions)

        return q
