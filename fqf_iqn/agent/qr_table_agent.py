import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

from utils import RunningMeanStats, LinearAnneaer
from torch.utils.tensorboard import SummaryWriter

from .table_base_agent import TableBaseAgent

class QRTableAgent(TableBaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=5*(10**7),
                 num_taus=32, lr=5e-5, gamma=0.99, multi_step=1, update_interval=4,
                 target_update_interval=10000, start_steps=50000, epsilon_train=0.01,
                 epsilon_eval=0.001, epsilon_decay_steps=250000, double_q_learning=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=27000, seed=0):
        super(QRTableAgent, self).__init__(
            env, test_env, log_dir, num_steps, lr, gamma, multi_step, 
            update_interval, target_update_interval, start_steps, epsilon_train, 
            epsilon_eval, epsilon_decay_steps, double_q_learning, log_interval, 
            eval_interval, num_eval_steps, max_episode_steps, seed)

        self.num_taus = num_taus

        # Online network.
        self.online_net = np.zeros((
            self.env.nrow * self.env.ncol,
            self.num_taus, 4))
        # Target network.
        self.target_net = self.online_net.copy()

        # Copy parameters of the learning network to the target network.
        self.update_target()

    def exploit(self, state):
        # Act without randomness.
        action = self.online_net[state.argmax()].sum(axis=0).argmax()
        return action

    def learn(self, state, action, reward, next_state, done):
        self.learning_steps += 1
        # td_error = (reward + (1-done) * self.gamma * \
        #                 max(self.target_net[next_state.argmax()]) - self.online_net[state.argmax(), action])

        # self.online_net[state.argmax(), action] += self.lr * td_error

        for _ in range(int(self.lr*self.num_taus)):
            p = np.random.randint(0, self.num_taus)
            q = np.random.randint(0, self.num_taus)
            # print(self.target_net[next_state.argmax()].max(axis=1)[q])
            next_action = \
                self.online_net[state.argmax()].sum(axis=0).argmax()
            td_error = reward + (1-done) * self.gamma \
                * self.target_net[next_state.argmax(), :, next_action][q] \
                - self.online_net[state.argmax(), p, action]

            self.online_net[state.argmax(), p, action] += self.lr*td_error

