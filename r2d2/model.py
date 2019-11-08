import os
import torch
import torch.nn as nn
import fasteners

from common.model import AbstractModel, Flatten,\
    weights_init_xavier, weights_init_he

class LSTMNetwork(AbstractModel):
    def __init__(self, n_state, n_act):
        super(LSTMNetwork, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.input_channel = n_state
        self.n_act = n_act

        self.hx = None
        self.cx = None

        self.vis_layers = nn.Sequential(
            # (84, 84, 1) -> (20, 20, 16)
            nn.Conv2d(self.input_channel, 32, kernel_size=8, stride=4),
            nn.ReLU(True),
            # (20, 20, 32) -> (9, 9, 32)
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(True),
            # (9, 9, 32) -> (7, 7, 64)
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(True),
            Flatten()
        ).apply(weights_init_he)

        self.common_fc = nn.Sequential(
            nn.Linear(7*7*64, 512).apply(weights_init_he),
            nn.ReLU(True),
        )

        self.lstm = nn.LSTMCell(512, 512).apply(weights_init_xavier)

        self.common_fc2 = nn.Sequential(
            nn.Linear(512, 256).apply(weights_init_he),
            nn.ReLU(True),
        )

        self.V_fc = nn.Sequential(
            nn.Linear(256, 1).apply(weights_init_xavier)
        )

        self.A_fc = nn.Sequential(
            nn.Linear(256, self.n_act).apply(weights_init_xavier)
        )

    def forward(self, vis, return_hs_cs=False):
        if len(vis.size()) == 5:
            batch_size, seq_size, _, _, _ = vis.size()
        else:
            batch_size = seq_size = 1

        vis = vis.contiguous().view(-1, self.input_channel, 84, 84)

        hs = self.vis_layers(vis)

        hs = hs.view(batch_size, seq_size, 7 * 7 * 64)

        hs = self.common_fc(hs)

        hs_seq = []
        cs_seq = []
        for i in range(hs.shape[1]):
            h = hs[:, i]
            self.hx, self.cx = self.lstm(h, (self.hx, self.cx))
            hs_seq.append(self.hx)
            cs_seq.append(self.cx)
        hx = torch.stack(hs_seq, dim=1)
        cx = torch.stack(cs_seq, dim=1)

        # vec = vec.view(batch_size, seq_size, self.vector_dim)
        # hs = torch.cat([hx, vec], dim=2)

        hs = self.common_fc2(hs)

        # V
        V = self.V_fc(hs)
        # A
        A = self.A_fc(hs)
        # Q
        Q = V + A - A.mean(2, keepdim=True)

        if return_hs_cs:
            return Q, hx, cx

        return Q

    def set_recc(self, hx, cx):
        self.hx = hx
        self.cx = cx

    def get_recc(self):
        hx = self.hx.clone().detach()
        cx = self.cx.clone().detach()
        arr = torch.stack([hx, cx], dim=0)
        return arr

    def reset_recc(self):
        self.hx = torch.zeros((1, 512), device=self.device, dtype=torch.float)
        self.cx = torch.zeros((1, 512), device=self.device, dtype=torch.float)


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
