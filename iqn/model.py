import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvNet(nn.Module):
    def __init__(self, n_state, n_actions, n_quant):
        super(ConvNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.n_state = n_state
        self.n_actions = n_actions
        self.n_quant = n_quant

        self.phi = nn.Linear(64, 64)
        # self.phi_bias = nn.Parameter(torch.zeros(64))

        self.fc0 = nn.Linear(n_state, 64)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_q = nn.Linear(64, n_actions)

    def forward(self, x):
        mb_size = x.size(0)

        # [batch_size, quant, 4]
        x = x.repeat(1, 1, self.n_quant).view(mb_size, self.n_quant, self.n_state)
        x = x.view(mb_size, -1, self.n_state)
        # [batch_size, quant, 64]
        x = F.relu(self.fc0(x))

        # [batch_size, quant, 1]
        tau = torch.rand(mb_size, self.n_quant).to(self.device).view(mb_size, -1, 1)

        # [1, 64]
        pi_mtx = math.pi * torch.arange(0, 64, 1.0).to(self.device).unsqueeze(0)
        # [batch_size, quant, 64]
        cos_tau = torch.cos(torch.matmul(tau, pi_mtx))
        # [batch_size, quant, quantile_embedding_dim]
        phi = F.relu(self.phi(cos_tau))

        # [batch_size, quant, 4]
        h = x * phi

        # [batch_size, quant, 64]
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))

        # [batch_size, quant, 2]
        action_value = self.fc_q(h).view(mb_size, self.n_quant, self.n_actions)
        # action_value = action_value.transpose(0, 2, 1)

        return action_value, tau
