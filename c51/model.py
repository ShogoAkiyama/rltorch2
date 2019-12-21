import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, n_state, n_actions, n_atom):
        super(ConvNet, self).__init__()

        self.n_actions = n_actions
        self.n_atom = n_atom

        self.fc1 = nn.Linear(n_state, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_q = nn.Linear(64, n_actions * n_atom)

    def forward(self, x):
        mb_size = x.size(0)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))

        # note that output of C-51 is prob mass of value distribution
        action_value = F.softmax(self.fc_q(h).view(mb_size, self.n_actions, self.n_atom), dim=2)

        return action_value
