import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def initialize_weights_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class DQNBase(nn.Module):

    def __init__(self, num_channels, embedding_dim=64):
        super(DQNBase, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(num_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        ).apply(initialize_weights_he)

        self.embedding_dim = embedding_dim

    def forward(self, states):
        batch_size = states.shape[0]

        # Calculate embeddings of states.
        state_embedding = self.net(states)
        assert state_embedding.shape == (batch_size, self.embedding_dim)

        return state_embedding


class OutLinearNetwork(nn.Module):

    def __init__(self, num_actions, embedding_dim=64):
        super(OutLinearNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, num_actions),
        )

        self.num_actions = num_actions
        self.embedding_dim = embedding_dim

    def forward(self, states_embeddings):
        batch_size = states_embeddings.shape[0]

        # Calculate embeddings of states.
        q = self.net(states_embeddings)
        assert q.shape == (batch_size, self.num_actions)

        return q


class FractionProposalNetwork(nn.Module):

    def __init__(self, num_taus=32, embedding_dim=64):
        super(FractionProposalNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, num_taus)
        ).apply(lambda x: initialize_weights_xavier(x, gain=0.01))

        self.num_taus = num_taus
        self.embedding_dim = embedding_dim

    def forward(self, state_embeddings):

        batch_size = state_embeddings.shape[0]

        # Calculate (log of) probabilities q_i in the paper.
        log_probs = F.log_softmax(self.net(state_embeddings), dim=1)
        probs = log_probs.exp()
        assert probs.shape == (batch_size, self.num_taus)

        tau_0 = torch.zeros(
            (batch_size, 1), dtype=state_embeddings.dtype,
            device=state_embeddings.device)
        taus_1_N = torch.cumsum(probs, dim=1)

        # Calculate \tau_i (i=0,...,N).
        taus = torch.cat((tau_0, taus_1_N), dim=1)
        assert taus.shape == (batch_size, self.num_taus+1)

        # Calculate \hat \tau_i (i=0,...,N-1).
        tau_hats = (taus[:, :-1] + taus[:, 1:]) / 2.
        assert tau_hats.shape == (batch_size, self.num_taus)

        # Calculate entropies of value distributions.
        entropies = -(log_probs * probs).sum(dim=-1, keepdim=True)
        assert entropies.shape == (batch_size, 1)

        return taus, tau_hats, entropies


class CosineEmbeddingNetwork(nn.Module):

    def __init__(self, num_cosines=64, embedding_dim=32, noisy_net=False):
        super(CosineEmbeddingNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.linear(num_cosines, embedding_dim),
            nn.ReLU()
        )
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def forward(self, taus):
        batch_size = taus.shape[0]
        num_taus = taus.shape[1]

        # Calculate i * \pi (i=1,...,N).
        i_pi = np.pi * torch.arange(
            start=1, end=self.num_cosines+1, dtype=taus.dtype,
            device=taus.device).view(1, 1, self.num_cosines)

        # Calculate cos(i * \pi * \tau).
        cosines = torch.cos(
            taus.view(batch_size, num_taus, 1) * i_pi
            ).view(batch_size * num_taus, self.num_cosines)

        # Calculate embeddings of taus.
        tau_embeddings = self.net(cosines).view(
            batch_size, num_taus, self.embedding_dim)

        return tau_embeddings


class QuantileNetwork(nn.Module):

    def __init__(self, num_actions, embedding_dim=64, dueling_net=False,
                 noisy_net=False):
        super(QuantileNetwork, self).__init__()

        if not dueling_net:
            self.net = nn.Sequential(
                nn.linear(embedding_dim, 512),
                nn.ReLU(),
                linear(512, num_actions),
            )
        else:
            self.advantage_net = nn.Sequential(
                nn.linear(embedding_dim, 512),
                nn.ReLU(),
                nn.linear(512, num_actions),
            )
            self.baseline_net = nn.Sequential(
                nn.linear(embedding_dim, 512),
                nn.ReLU(),
                nn.linear(512, 1),
            )

        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net

    def forward(self, state_embeddings, tau_embeddings):
        assert state_embeddings.shape[0] == tau_embeddings.shape[0]
        assert state_embeddings.shape[1] == tau_embeddings.shape[2]

        # NOTE: Because variable taus correspond to either \tau or \hat \tau
        # in the paper, num_taus isn't neccesarily the same as fqf_iqn.num_taus.
        batch_size = state_embeddings.shape[0]
        num_taus = tau_embeddings.shape[1]

        # Reshape into (batch_size, 1, embedding_dim).
        state_embeddings = state_embeddings.view(
            batch_size, 1, self.embedding_dim)

        # Calculate embeddings of states and taus.
        embeddings = (state_embeddings * tau_embeddings).view(
            batch_size * num_taus, self.embedding_dim)

        # Calculate quantile values.
        if not self.dueling_net:
            quantiles = self.net(embeddings)
        else:
            advantages = self.advantage_net(embeddings)
            baselines = self.baseline_net(embeddings)
            quantiles =\
                baselines + advantages - advantages.mean(1, keepdim=True)

        return quantiles.view(batch_size, num_taus, self.num_actions)

