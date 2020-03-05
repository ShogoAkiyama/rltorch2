import torch
from torch import nn

from fqf_iqn.network import DQNBase, OutLinearNetwork


class DQN(nn.Module):

    def __init__(self, num_channels, num_actions):
        super(DQN, self).__init__()

        # Feature extractor of DQN.
        self.dqn_net = DQNBase(num_channels=num_channels)
        self.out_net = OutLinearNetwork(num_actions)

        self.num_channels = num_channels
        self.num_actions = num_actions

    def calculate_state_embeddings(self, states):
        return self.dqn_net(states)

    def calculate_quantiles(self, taus, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        tau_embeddings = self.cosine_net(taus)
        return self.quantile_net(state_embeddings, tau_embeddings)

    def calculate_q(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None\
            else state_embeddings.shape[0]

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        # Calculate expectations of value distributions.
        q = self.out_net(state_embeddings)
        assert q.shape == (batch_size, self.num_actions)

        return q
