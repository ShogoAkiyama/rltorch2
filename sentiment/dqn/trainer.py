import numpy as np

import torch.optim as optim
import torch

from model import DQN


class Trainer:
    def __init__(self, args, TEXT, train_dl):

        self.train_dl = train_dl

        self.target_update_freq = args.target_update_freq
        self.device = args.device
        self.gamma = args.gamma

        self.epochs = 0
        self.epoch_loss = 0

        vocab_size = len(TEXT.vocab.freqs)
        self.model = DQN(TEXT.vocab.vectors,
                         vocab_size, args.embedding_dim, args.n_filters,
                         args.filter_sizes, args.pad_idx).to(args.device)
        self.target_model = DQN(TEXT.vocab.vectors,
                                vocab_size, args.embedding_dim, args.n_filters,
                                args.filter_sizes, args.pad_idx).to(args.device)

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

            if self.epochs % 10 == 0:
                self.evaluation()

            self.log()

    def train_episode(self):

        for batch in self.train_dl:
            # curr_q
            states = batch.Text1[0].to(self.device)
            next_states = batch.Text2[0].to(self.device)
            rewards = batch.Label.to(self.device)

            self.train(states, next_states, rewards)

    def train(self, states, next_states, rewards):
        curr_q = self.model(states)

        with torch.no_grad():
            next_actions = torch.argmax(self.model(next_states), 1).view(-1, 1)

            next_q = self.target_model(next_states).gather(1, next_actions).expand(-1, 2)
            rewards = torch.cat((torch.zeros(len(rewards), 1).to(self.device),
                                 rewards.view(-1, 1)), 1)
            target_q = rewards + (self.gamma * next_q)

        loss = torch.mean((curr_q - target_q)**2)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.epoch_loss += loss.item()

    def evaluation(self):
        epi_rewards = 0
        for batch in self.train_dl:
            states = batch.Text1[0].to(self.device)
            rewards = batch.Label.to(self.device)

            with torch.no_grad():
                actions = torch.argmax(self.model(states), 1)

            epi_rewards += (actions * rewards).detach().cpu().numpy().sum()

        print(' '*20,
              'train_reward: ', epi_rewards)

    def log(self):
        print('epoch: ', self.epochs,
              ' loss: {:.3f}'.format(self.epoch_loss))
