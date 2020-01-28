import os
import numpy as np
from glob import glob

import torch.optim as optim
import torch

from model import QRDQN

LOG_DIR = os.path.join('.', 'logs')


class Trainer:
    def __init__(self, args, TEXT, train_dl):

        self.train_dl = train_dl

        self.target_update_freq = args.target_update_freq
        self.evaluation_freq = args.evaluation_freq
        self.network_save_freq = args.network_save_freq

        self.device = args.device
        self.gamma = args.gamma
        self.num_actions = args.num_actions

        # quantile
        self.num_quantile = args.num_quantile
        self.quantile_weight = 1.0 / self.num_quantile
        self.cumulative_density = torch.FloatTensor(
            (2 * np.arange(self.num_quantile) + 1) / (2.0 * self.num_quantile)
        ).to(self.device)

        self.epochs = 0
        self.epoch_loss = 0

        vocab_size = len(TEXT.vocab.freqs)
        self.model = QRDQN(vocab_size, args.embedding_dim, args.n_filters,
                           args.filter_sizes, args.pad_idx, args.num_actions).to(args.device)
        self.target_model = QRDQN(vocab_size, args.embedding_dim, args.n_filters,
                                args.filter_sizes, args.pad_idx, args.num_actions).to(args.device)

        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)

    def run(self):
        while True:
            self.epoch_loss = 0
            self.epochs += 1
            self.train_episode()

            # update target_model
            if self.epochs % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            if self.epochs % self.evaluation_freq == 0:
                self.evaluation(self.train_dl)

            if self.epochs % self.network_save_freq == 0:
                self.save_model()

            self.log()

    def train_episode(self):
        for batch in self.train_dl:
            states = batch.Text1[0].to(self.device)
            next_states = batch.Text2[0].to(self.device)
            rewards = batch.Label.to(self.device)

            self.train(states, next_states, rewards)

    def train(self, states, next_states, rewards):
        curr_q = self.model(states).view(-1, self.num_quantile)

        # target_q
        with torch.no_grad():
            next_dist = self.model(next_states) * self.quantile_weight
            next_action = next_dist.sum(dim=2).max(1)[1].view(next_states.size(0), 1, 1).expand(
                -1, -1, self.num_quantile)

            next_q = self.target_model(next_states).gather(1, next_action)
            next_q = next_q.expand(-1, self.num_actions, -1).reshape(-1, self.num_quantile)

            if self.num_actions >= 2:
                rewards = torch.cat((torch.zeros(len(rewards), 1).to(self.device),
                                     rewards.view(-1, 1)), 1).view(-1, 1)
            else:
                rewards = rewards.view(-1 ,1)

            target_q = rewards + (self.gamma * next_q)

        diff = target_q.t().unsqueeze(-1) - curr_q.unsqueeze(0)

        loss = self.huber(diff) * torch.abs(
            self.cumulative_density.view(1, -1) - (diff < 0).to(torch.float))

        loss = loss.transpose(0, 1)
        loss = loss.mean(1).sum(-1).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.epoch_loss += loss.item()

    def evaluation(self, dl):
        epi_rewards = 0
        dist_hist = []
        rewards_hist = []

        for batch in dl:
            states = batch.Text1[0].to(self.device)
            rewards = batch.Label.to(self.device)

            with torch.no_grad():
                dist = self.model(states) * self.quantile_weight
                dist_hist.append(dist.cpu().detach().numpy())
                rewards_hist.append(rewards.cpu().detach().numpy())
                if self.num_actions >=2:
                    actions = dist.sum(dim=2).max(1)[1]
                else:
                    _sum = dist.sum(dim=2)
                    actions = torch.where(
                                _sum > 0, 
                                torch.LongTensor([1]).to(self.device),
                                torch.LongTensor([0]).to(self.device))

            epi_rewards += (actions * rewards).detach().cpu().numpy().sum()

        print(' '*20,
              'train_reward: ', epi_rewards)
        
        return dist_hist, rewards_hist

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)

    def log(self):
        print('epoch: ', self.epochs,
              ' loss: {:.3f}'.format(self.epoch_loss))

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(LOG_DIR, str(self.epochs))+'.pt')

    def load_model(self):
        model_path = sorted(glob(os.path.join(LOG_DIR, '*')))[-1]
        self.model.load_state_dict(torch.load(model_path))
    