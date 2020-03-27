import torch
import torch.nn as nn


class AbstractModel(nn.Module):
    def __init__(self):
        super(AbstractModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, vis, vec):
        raise Exception("You need to implement forward() function.")

    def load(self):
        model_path = './model.pt'
        load_model = torch.load(model_path, map_location=self.device)
        self.load_state_dict(load_model)

    @property
    def n_params(self):
        n = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            n += nn
        return n


class QNetwork(AbstractModel):
    def __init__(self, n_state, n_act):
        super(QNetwork, self).__init__()

        self.input_channel = n_state
        self.n_act = n_act

        self.vis_layers = nn.Sequential(
            # (84, 84, 1) -> (20, 20, 16)
            nn.Conv2d(self.input_channel, 32, kernel_size=8, stride=4),
            nn.ReLU(True),
            # (20, 20, 32) -> (9, 9, 64)
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(True),
            Flatten()
        ).apply(weights_init_he)

        self.common_fc = nn.Sequential(
            nn.Linear(9 * 9 * 64, 512).apply(weights_init_he),
            nn.ReLU(True),
        )

        self.V_fc = nn.Sequential(
            nn.Linear(512, 256).apply(weights_init_he),
            nn.ReLU(True),
            nn.Linear(256, 1).apply(weights_init_xavier)
        )

        self.A_fc = nn.Sequential(
            nn.Linear(512, 256).apply(weights_init_he),
            nn.ReLU(True),
            nn.Linear(256, self.n_act).apply(weights_init_xavier)
        )

    def forward(self, vis):
        hs = self.vis_layers(vis)
        hs = self.common_fc(hs)

        # V
        V = self.V_fc(hs)

        # A
        A = self.A_fc(hs)

        # Q
        Q = V + A - A.mean(1, keepdim=True)

        return Q


def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


def weights_init_he(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0)
        nn.init.constant_(m.bias, 0)


def weights_init_lstm(m):
    if isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)



