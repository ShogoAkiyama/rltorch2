import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvNet(nn.Module):
    def __init__(self, n_state, n_actions, n_quant):
        super(ConvNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.n_actions = n_actions
        self.n_quant = n_quant

        self.phi = nn.Linear(1, n_state, bias=False)
        self.phi_bias = nn.Parameter(torch.zeros(n_state))

        self.fc1 = nn.Linear(n_state, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_q = nn.Linear(64, n_actions)

    def forward(self, x):
        mb_size = x.size(0)

        tau = torch.rand(self.n_quant, 1).to(self.device)
        quants = torch.arange(0, 64, 1.0).to(self.device)
        pi_mtx = tau * quants
        cos_trans = torch.cos(pi_mtx * math.pi).unsqueeze(2)  # (N_QUANT, N_QUANT, 1)
        phi = F.relu(self.phi(cos_trans).mean(dim=1) + self.phi_bias.unsqueeze(0)).unsqueeze(0)

        x = x.view(mb_size, -1).unsqueeze(1)
        x = x * phi  # (m, N_QUANT, 4)

        h = F.relu(self.fc1(x))   # (m, N_QUANT, 64)
        h = F.relu(self.fc2(h))

        action_value = self.fc_q(h).transpose(1, 2)

        return action_value, tau
