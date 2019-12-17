import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    def __init__(self, ob_space, ac_space):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(ob_space, 32)   # 入力層
        self.fc2 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, ac_space)

    def forward(self, ob):
        h = F.relu(self.fc1(ob))
        h = F.relu(self.fc2(h))
        h = self.output_layer(h)
        h = F.log_softmax(h, dim=-1)
        return h

class CriticNetwork(nn.Module):   # 行動価値を出力する
    def __init__(self, ob_space):
      super(CriticNetwork, self).__init__()
      self.fc1 = nn.Linear(ob_space, 32)   # 行動を加えることで行動価値を出力
      self.fc2 = nn.Linear(32, 32)
      self.output_layer = nn.Linear(32, 1)

    def forward(self, ob):
      h = ob
      h = F.relu(self.fc1(h))
      h = F.relu(self.fc2(h))
      return self.output_layer(h)
