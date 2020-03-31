import torch
from torch import nn


from fqf_iqn.network import CosineEmbeddingNetwork, QuantileNetwork

class IQN(nn.Module):

    def __init__(self, num_states, num_actions, K=32, num_cosines=32,
                 embedding_dim=512, c=0, sensitive=False):
        super(IQN, self).__init__()

        # Quantile network.
        self.q_net = nn.Sequential(
            nn.Linear(num_states, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        # Cosine embedding network.
        self.cosine_net = CosineEmbeddingNetwork(
            num_cosines=num_cosines, embedding_dim=embedding_dim)
        # Quantile network.
        self.quantile_net = QuantileNetwork(
            num_actions=num_actions, embedding_dim=embedding_dim)

        self.K = K
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

        self.sensitive = sensitive
        self.c = c

    def calculate_state_embeddings(self, states):
        return self.q_net(states)

    def calculate_quantiles(self, taus, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None

        tau_embeddings = self.cosine_net(taus)
        return self.quantile_net(state_embeddings, tau_embeddings)

    def calculate_q(self, states=None, state_embeddings=None, beta=False):
        batch_size = states.shape[0] if states is not None \
            else state_embeddings.shape[0]

        if state_embeddings is None:
            state_embeddings = self.q_net(states)

        # Sample fractions.
        taus = torch.rand(
            batch_size, self.K, dtype=torch.float32,
            device=state_embeddings.device)

        if self.c == 0:
            taus = taus
        elif self.sensitive:
            taus *= self.c
        else:
            taus = (1 - self.c) * taus + (1 - self.c)

        # Calculate quantiles.
        quantiles = self.calculate_quantiles(
            taus, state_embeddings=state_embeddings)
        assert quantiles.shape == (batch_size, self.K, self.num_actions)

        # Calculate expectations of value distributions.
        q = quantiles.mean(dim=1)
        assert q.shape == (batch_size, self.num_actions)

        return q
