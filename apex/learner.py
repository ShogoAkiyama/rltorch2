import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .model import QNetwork

from common.utils import grad_false
from common.abstract import Abstract
from common.replay_memory import EpisodeReplayMemory


def learner_process(args, shared_memory, shared_weights):
    learner = Learner(args, shared_memory, shared_weights)
    learner.learner_run()

class Learner(Abstract):
    def __init__(self, args, shared_memory, shared_weights):
        super(Learner, self).__init__(args, shared_memory, shared_weights)

        self.summary_path = os.path.join(
            './', 'logs', 'summary', 'leaner')
        self.writer = SummaryWriter(log_dir=self.summary_path)

        # param
        self.train_steps = 0
        self.epochs = 0

        self.net = QNetwork(self.n_state, self.n_act)
        self.target_net = QNetwork(self.n_state, self.n_act)

        self.net.to(self.device)
        self.target_net.to(self.device)

        # target network
        self.target_net.eval().apply(grad_false)
        self.target_net.load_state_dict(self.net.state_dict())

        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.optim_lr)

        # model save
        self.save_model()

        # memory
        self.memory = EpisodeReplayMemory(args)

    def train(self):
        self.total_loss = 0
        for epoch in range(self.update_per_epoch):
            self.train_steps += 1
            # sample batch
            batch, seq_idx, epi_idx, weights = \
                self.memory.get_batch()

            # process batch
            batch = self.process_batch(batch)
            weights = torch.tensor(weights, device=self.device, dtype=torch.float).view(-1, 1)

            # calculate Q and target
            curr_q, target_q = self.q_value(batch)

            # update model
            loss = torch.mean((target_q - curr_q) ** 2 * weights)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update priority
            td_loss = np.abs(target_q.cpu() - curr_q.detach().cpu()).numpy()
            self.memory.update_priority(td_loss, seq_idx, epi_idx)

            self.total_loss += loss.item()

    def q_value(self, batch):
        # curr Q
        batch['action'] = batch['action'].view(-1, 1)
        curr_q = self.net(batch['state']).gather(1, batch['action'])

        with torch.no_grad():
            next_action = torch.argmax(
                self.net(batch['next_state']), 1).view(-1, 1)
            next_q = self.target_net(
                batch['next_state']).gather(1, next_action)

        # target Q
        target_q = batch['reward'].view(-1, 1) + \
                   (self.gamma ** self.multi_step) * next_q * (1.0 - batch['done'].view(-1, 1))

        return curr_q, target_q
