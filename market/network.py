import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_shape, num_actions, quantiles=51):
        super(Network, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.quantiles = quantiles

        self.fc0 = nn.Linear(self.input_shape, 32)

        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, self.num_actions * self.quantiles)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x.view(-1, self.num_actions, self.quantiles)
