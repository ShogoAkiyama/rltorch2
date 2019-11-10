import os
import torch
import torch.nn as nn
import fasteners

from common.model import AbstractModel, Flatten,\
    weights_init_xavier, weights_init_he

class QNetwork(AbstractModel):
    def __init__(self, n_state, n_act):
        super(QNetwork, self).__init__()

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

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

    def save(self, path, target=False):
        model_path = os.path.join(path, 'target_model.pt') \
            if target else os.path.join(path, 'model.pt')

        lock = fasteners.InterProcessLock(model_path)
        gotten = lock.acquire(blocking=False)
        try:
            if gotten:
                torch.save(self.state_dict(), model_path)

        finally:
            if gotten:
                lock.release()
