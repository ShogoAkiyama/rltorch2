import os
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from memory import LazyMultiStepMemory
from utils import RunningMeanStats, LinearAnneaer


class TableBaseAgent:

    def __init__(self, env, test_env, log_dir, num_steps=5*(10**7),
                 lr=5e-5, gamma=0.99, multi_step=1, update_interval=4,
                 target_update_interval=10000, start_steps=50000, epsilon_train=0.01,
                 epsilon_eval=0.001, epsilon_decay_steps=250000, double_q_learning=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=27000, seed=0):

        self.env = env
        self.test_env = test_env

        self.start_steps = start_steps
        self.max_episode_steps = max_episode_steps
        self.epsilon_train = LinearAnneaer(
            1.0, epsilon_train, epsilon_decay_steps)
        self.epsilon_eval = epsilon_eval
        self.num_steps = num_steps
        self.num_eval_steps = num_eval_steps
        self.lr = lr
        self.gamma = gamma

        self.target_update_interval = target_update_interval
        self.log_interval = log_interval
        self.update_interval = update_interval
        self.eval_interval = eval_interval
        self.summary_dir = os.path.join(log_dir, 'summary')

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_return = RunningMeanStats(log_interval)

        self.steps = 0
        self.episodes = 0
        self.learning_steps = 0

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return self.steps % self.update_interval == 0\
            and self.steps >= self.start_steps

    def is_greedy(self, eval=False):
        if eval:
            return np.random.rand() < self.epsilon_eval
        else:
            return self.steps < self.start_steps\
                or np.random.rand() < self.epsilon_train.get()

    def update_target(self):
        self.target_net = self.online_net.copy()

    def explore(self):
        # Act with randomness.
        action = self.env.action_space.sample()
        return action

    def exploit(self, state):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(
            self.online_net.state_dict(),
            os.path.join(save_dir, 'online_net.pth'))
        torch.save(
            self.target_net.state_dict(),
            os.path.join(save_dir, 'target_net.pth'))

    def load_models(self, save_dir):
        self.online_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'online_net.pth')))
        self.target_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'target_net.pth')))

    def train_episode(self):
   
        self.episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = False
        state = self.env.reset()

        while (not done) and episode_steps <= self.max_episode_steps:

            if self.is_greedy(eval=False):
                action = self.explore()
            else:
                action = self.exploit(state)

            next_state, reward, done, _ = self.env.step(action)

            self.learn(state, action, reward, next_state, done)

            self.steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state

            self.train_step_interval()

        # We log running mean of stats.
        self.train_return.append(episode_return)

        # We log evaluation results along with training frames = 4 * steps.
        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'return/train', self.train_return.get(), 4 * self.steps)

        if self.episodes % 1000 == 0:
            print(f'Episode: {self.episodes:<4}  '
                  f'episode steps: {episode_steps:<4}  '
                  f'return: {episode_return:<5.1f}')

    def train_step_interval(self):
        self.epsilon_train.step()

        if self.steps % self.target_update_interval == 0:
            self.update_target()

        if self.steps % self.eval_interval == 0:
            self.evaluate()

    def evaluate(self):
        num_episodes = 0
        num_steps = 0
        total_return = 0.0

        while True:
            state = self.test_env.reset()
            episode_steps = 0
            episode_return = 0.0
            done = False
            while (not done) and episode_steps <= self.max_episode_steps:
                if self.is_greedy(eval=True):
                    action = self.explore()
                else:
                    action = self.exploit(state)

                next_state, reward, done, _ = self.test_env.step(action)
                num_steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

            num_episodes += 1
            total_return += episode_return

            if num_steps > self.num_eval_steps:
                break

        # print(self.online_net.copy().reshape(self.env.nrow, self.env.ncol, 4)[0][1])

    def plot(self, q_value):
        state_size = 3
        q_nrow = self.env.nrow * state_size
        q_ncol = self.env.ncol * state_size

        # q_value = self.online_net.copy().reshape(self.env.nrow, self.env.ncol, 4)

        value = np.zeros((q_nrow, q_ncol))

        # Left, Down, Right, Up, Center
        value[1::3, ::3] += q_value[:, :, 0]
        value[2::3, 1::3] += q_value[:, :, 1]
        value[1::3, 2::3] += q_value[:, :, 2]
        value[::3, 1::3] += q_value[:, :, 3]
        value[1::3, 1::3] += q_value.mean(axis=2)

        # ヒートマップ表示
        fig = plt.figure(figsize=(6, 12))
        ax = fig.add_subplot(1, 1, 1)
        mappable0 = plt.imshow(value, cmap=cm.jet, interpolation="bilinear",
           vmax=abs(value).max(), vmin=-abs(value).max())
        ax.set_xticks(np.arange(-0.5, q_ncol, 3))
        ax.set_yticks(np.arange(-0.5, q_nrow, 3))
        ax.set_xticklabels(range(self.env.ncol + 1))
        ax.set_yticklabels(range(self.env.nrow + 1))
        ax.grid(which="both")

        # Start: green, Goal: blue, Hole: red
        ax.plot([1], [1],  marker="o", color='g', markersize=40, alpha=0.8)
        ax.plot([1], [4],  marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
        ax.plot([1], [7],  marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
        ax.plot([1], [10], marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
        ax.plot([1], [13], marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
        ax.plot([1], [16], marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
        ax.plot([1], [19], marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
        ax.plot([1], [22], marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
        ax.plot([1], [25], marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
        ax.plot([1], [28], marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
        ax.plot([1], [31], marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
        ax.plot([1], [34], marker="o", color='r', markersize=40, alpha=0.8)
        ax.text(1, 1.3, 'START', ha='center', size=12, c='w')
        ax.text(1, 34.3, 'GOAL', ha='center', size=12, c='w')

        fig.colorbar(mappable0, ax=ax, orientation="vertical")

        plt.show()

    def __del__(self):
        self.env.close()
        self.test_env.close()
        self.writer.close()
