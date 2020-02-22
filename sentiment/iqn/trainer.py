import os
import numpy as np
from glob import glob

import torch.optim as optim
import torch

from model import IQN

from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, args, text_vectors, vocab_size, train_dl):

        self.log_path = os.path.join('.', 'logs',
                            str(args.num_quantile) + '_' + str(args.gamma))
        self.summary_path = os.path.join(self.log_path, 'summary')

        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)

        self.writer = SummaryWriter(log_dir=self.summary_path)

        self.train_dl = train_dl

        self.target_update_freq = args.target_update_freq
        self.evaluation_freq = args.evaluation_freq
        self.network_save_freq = args.network_save_freq

        self.device = args.device
        self.gamma = args.gamma
        self.num_actions = args.num_actions

        # quantile
        self.num_quantile = args.num_quantile

        self.epochs = 0
        self.epoch_loss = 0
        self.epi_rewards = 0

        # vocab_size = len(TEXT.vocab.freqs)
        self.model = IQN(text_vectors, vocab_size, args.embedding_dim, args.n_filters,
                         args.filter_sizes, args.pad_idx,
                         n_actions=args.num_actions,
                         n_quant=self.num_quantile,
                         rnn=args.rnn).to(args.device)
        self.target_model = IQN(text_vectors, vocab_size, args.embedding_dim, args.n_filters,
                                args.filter_sizes, args.pad_idx,
                                n_actions=args.num_actions,
                                n_quant=self.num_quantile,
                                rnn=args.rnn).to(args.device)

        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 1000)

    def run(self):
        while True:
            self.epoch_loss = 0
            self.epochs += 1            
            self.model.train()

            self.scheduler.step()
            if self.epochs % 10 == 0:
                print(self.scheduler.get_lr()[0])

            self.train_episode()

            # update target_model
            if self.epochs % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            if self.epochs % self.evaluation_freq == 0:
                self.model.eval()
                self.evaluation(self.train_dl)

            if self.epochs % self.network_save_freq == 0:
                self.save_model()

            self.log()

    def train_episode(self):
        for batch in self.train_dl:
            states = batch['State'].to(self.device)
            next_states = batch['Next_State'].to(self.device)
            rewards = torch.round(batch['Reward'].to(self.device))

            self.train(states, next_states, rewards)

    def train(self, states, next_states, rewards):
        curr_q, tau, _ = self.model(states)
        # curr_q = curr_q.repeat(1, 1, self.num_quantile)

        # target_q
        with torch.no_grad():
            if self.gamma == 0.0:
                target_q = rewards.reshape(-1, 1)
                target_q = target_q.repeat(1, self.num_quantile)
            else:
                next_q, _, _ = self.target_model(next_states)
                next_q = next_q.squeeze(2)

                target_q = rewards.reshape(-1, 1) + self.gamma * next_q
            target_q = target_q.unsqueeze(1)
            # target_q = target_q.unsqueeze(2)
            # target_q = target_q.repeat(1, 1, self.num_quantile)
            # target_q = target_q.permute(0, 2, 1)

        # (BATCH, N_QUANT, N_QUANT)
        tau = tau.repeat(1, 1, self.num_quantile)

        diff = target_q - curr_q

        loss = self.huber(diff)

        I_delta = (diff<0).double()
        loss *= torch.abs(tau - I_delta).detach()

        # huber loss
        loss = torch.mean(torch.sum(torch.mean(loss, dim=2), dim=1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.model.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.epoch_loss += loss.item()

    def evaluation(self, dl):
        epi_rewards = 0
        dist_hist = []
        rewards_hist = []

        for batch in dl:
            states = batch['State'].to(self.device)
            rewards = batch['Reward'].to(self.device)

            with torch.no_grad():
                dist, _, _ = self.model(states)
                dist = dist.squeeze(2)
                dist_hist.append(dist.cpu().detach().numpy())
                rewards_hist.append(rewards.cpu().detach().numpy())
                # print(dist.shape)
                _mean = dist.sum(dim=1)
                actions = torch.where(
                            _mean > 0,
                            torch.LongTensor([1]).to(self.device),
                            torch.LongTensor([0]).to(self.device))
            # print(actions.shape, ' ', rewards.shape)
            epi_rewards += (actions * rewards).detach().cpu().numpy().sum()

        self.epi_rewards = epi_rewards

        print(' '*20,
              'train_reward: ', epi_rewards)
        
        return dist_hist, rewards_hist

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)

    def log(self):
        print('epoch: ', self.epochs,
              ' loss: {:.3f}'.format(self.epoch_loss))

        self.writer.add_scalar("loss", self.epoch_loss, self.epochs)
        self.writer.add_scalar("epi_rewards", self.epi_rewards, self.epochs)

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.log_path, str(self.epochs))+'.pt')

    def load_model(self):
        model_path = sorted(glob(os.path.join(self.log_path, '*.pt')))[-1]
        self.model.load_state_dict(torch.load(model_path))

    def __del__(self):
        self.writer.close()
