import torch
import torch.nn as nn
import os
import fasteners
import pickle


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


class AbstractModel(nn.Module):
    def __init__(self):
        super(AbstractModel, self).__init__()

    def forward(self, vis, vec):
        raise Exception("You need to implement forward() function.")

    def load(self, path, target=False):
        model_path = os.path.join(path, 'target_model.pt')\
            if target else os.path.join(path, 'model.pt')

        lock = fasteners.InterProcessLock(model_path)
        gotten = lock.acquire(blocking=False)
        try:
            if gotten and os.path.isfile(model_path):
                load_model = torch.load(model_path)
                self.load_state_dict(load_model)

        except pickle.UnpicklingError:
            print("Pickle data was truncated.")

        finally:
            if gotten:
                lock.release()

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

    @property
    def n_params(self):
        n = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            n += nn
        return n
