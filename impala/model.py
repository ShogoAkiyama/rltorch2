import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    def __init__(self, n_state, n_act):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(n_state, 32)
        self.fc2 = nn.Linear(32, 32)
        self.output_linear = nn.Linear(32, n_act)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = self.output_linear(h)
        h = F.log_softmax(h, dim=-1)
        return h


class CriticNetwork(nn.Module):
    def __init__(self, n_state):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(n_state, 32)
        self.fc2 = nn.Linear(32, 32)
        self.output_linear = nn.Linear(32, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = self.output_linear(h)
        return h
