import os
import torch
import numpy as np
import time
from collections import deque

from .replay_memory import EpisodeReplayMemory
from torch.utils.tensorboard import SummaryWriter

from .model import QNetwork
from .abstract import Abstract
from common.utils import grad_false
from common.logger import RewardLogger

import torch.optim as optim

class Trainer(Abstract):

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        self.eps_greedy = 1.0
        self.epochs = 0
        self.n_episodes = 0
        self.train_steps = 0

        self.writer = SummaryWriter(log_dir=self.summary_path)

        # reward logger
        self.reward_logger = RewardLogger()

        # memory
        self.memory = EpisodeReplayMemory(args)

        # network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = QNetwork(self.n_state, self.n_act)
        self.target_net = QNetwork(self.n_state, self.n_act).eval().apply(grad_false)

        self.net.to(self.device)
        self.target_net.to(self.device)

        self.optim_lr = args.lr
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.optim_lr)

        self.env_state = self.env.reset()

        self.key_list = ['state', 'action', 'reward', 'done', 'priority']

        self.actor_interval()

    def run(self):
        self.time = time.time()
        while True:
            self.n_steps += 1

            self.interact()

            if self.episode_done:
                if self.n_steps >= self.multi_step:
                    self.memory.add(self.episode_buff)

                self.reward_logger.add_reward(self.episode_reward)
                self.env_state = self.env.reset()

                self.actor_interval()

                if self.n_episodes >= self.batch_size:
                    self.train()
                    self.learner_interval()

    def train(self):
        self.total_loss = 0
        self.epochs += 1
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

    def interact(self):
        state = self.env_state
        action = self.select_action(state)
        done = self.env_done
        next_state, reward, next_done, _ = self.env.step(action)

        self.episode_reward += reward

        self.add_episode_buff(state, action, done)

        if next_done:
            self.actor_log()
            self.n_episodes += 1
            self.episode_done = True

            self.reward_deque.append(reward)

            while len(self.reward_deque) > 0:

                self.episode_buff['state'].append(np.zeros((int(self.n_state/4), 84, 84), dtype=np.uint8))

                # calc reward
                discount_reward = self.calc_discount_reward()
                self.episode_buff['reward'].append(discount_reward)
                self.episode_buff['done'].append(True)

                # calc priority
                self.calc_priority(discount_reward, done=True)

                self.reward_deque.popleft()

        else:
            self.env_state = next_state
            self.env_done = next_done

            self.reward_deque.append(reward)

            if self.n_steps >= self.multi_step:
                # calc reward
                discount_reward = self.calc_discount_reward()
                self.episode_buff['reward'].append(discount_reward)
                self.calc_priority(discount_reward)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        if self.eps_greedy > 0:
            self.eps_greedy -= 1e-5

        with torch.no_grad():
            curr_q = self.net(state).cpu().numpy().reshape(-1)
            action = int(curr_q.argmax())
            self.next_q = self.target_net(state).cpu().numpy().reshape(-1)[action]

        if np.random.rand() < self.eps_greedy:
            action = np.random.choice(self.n_act)

        self.curr_q_deque.append(curr_q[action])

        return action

    def calc_priority(self, discount_reward, done=False):
        curr_q = self.curr_q_deque.popleft()

        if done:
            target_q = discount_reward
        else:
            next_q = self.next_q
            target_q = discount_reward + (self.gamma ** self.multi_step) * next_q

        # add priority
        priority = np.abs(target_q - curr_q)
        priority = np.power(priority, self.priority_exp)
        self.episode_buff['priority'].append(priority)

    def actor_interval(self):
        # reset
        self.episode_done = False
        self.next_q = None
        self.env_done = False
        self.episode_reward = 0
        self.n_steps = 0
        self.reward_deque = deque(maxlen=self.multi_step)
        self.curr_q_deque = deque(maxlen=self.multi_step)

        self.episode_buff = {}
        for key in self.key_list:
            self.episode_buff[key] = list()

        if self.epochs % self.save_model_interval*100 == 0:
            self.net.save(self.model_path)
            self.target_net.save(self.model_path, target=True)

    def learner_interval(self):
        if self.epochs % self.eval_interval == 0:
            self.evaluate()

        if self.epochs % self.save_model_interval*100 == 0:
            self.net.save(self.model_path)
            self.target_net.save(self.model_path, target=True)

        if self.epochs % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.net.state_dict())
            self.target_net.eval()

        self.learner_log()

    def evaluate(self):
        episodes = 10
        returns = np.zeros((episodes,), dtype=np.float32)
        action_bar = np.zeros(self.n_act, np.int)

        for i in range(episodes):
            state = self.env.reset()
            episode_reward = 0.
            done = False
            while not done:
                action = self.exploit(state)
                next_state, reward, done, _ = self.env.step(action)
                action_bar[action] += 1
                episode_reward += reward
                state = next_state

            returns[i] = episode_reward

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        self.writer.add_scalar(
            'reward/test', mean_return, self.train_steps)
        print('Learer  '
              f'Num steps: {self.train_steps:<5} '
              f'reward: {mean_return:<5.1f}+/- {std_return:<5.1f}')

    def exploit(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.net(state).argmax().item()
        return action
