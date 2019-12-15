import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(nn.Module):
    def __init__(self, s_channel, a_space):
        super(ActorCritic, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.vis_layers = nn.Sequential(
            nn.Conv2d(s_channel, 32, kernel_size=8, stride=4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(True),
            Flatten()
        )

        self.lstm = nn.LSTMCell(9*9*64, 256)

        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, a_space)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        # inputs, (hx, cx) = inputs
        x = self.vis_layers(inputs)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        x = self.hx

        return self.critic_linear(x), F.log_softmax(self.actor_linear(x), dim=-1)

    def reset_recc(self, batch_size=1):
        self.cx = torch.zeros(batch_size, 256).to(self.device)
        self.hx = torch.zeros(batch_size, 256).to(self.device)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def weights_init_he(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0)
        nn.init.constant_(m.bias, 0)