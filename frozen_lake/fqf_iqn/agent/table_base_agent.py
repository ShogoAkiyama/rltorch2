import numpy as np
import seaborn as sns
from pylab import *

import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

from torch.utils.tensorboard import SummaryWriter

from utils import RunningMeanStats, LinearAnneaer


class TableBaseAgent:

    def __init__(self, env, test_env, log_dir, num_steps=5 * (10 ** 7),
                 lr=5e-5, gamma=0.99, multi_step=1, update_interval=4,
                 target_update_interval=10000, start_steps=50000, epsilon_train=0.01,
                 epsilon_eval=0.001, epsilon_decay_steps=250000, double_q_learning=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=27000, seed=0, cuda=True):

        self.env = env
        self.test_env = test_env
        self.num_actions = env.num_actions

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.start_steps = start_steps
        self.max_episode_steps = max_episode_steps
        self.epsilon_train = LinearAnneaer(
            1.0, epsilon_train, epsilon_decay_steps)
        self.epsilon_eval = epsilon_eval
        self.num_steps = num_steps
        self.num_eval_steps = num_eval_steps
        self.lr = lr
        self.gamma = gamma
        self.double_q_learning = double_q_learning

        self.target_update_interval = target_update_interval
        self.log_interval = log_interval
        self.update_interval = update_interval
        self.eval_interval = eval_interval
        self.summary_dir = os.path.join(log_dir, 'summary')

        self.log_dir = log_dir
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
        return self.steps % self.update_interval == 0 \
               and self.steps >= self.start_steps

    def is_greedy(self, eval=False):
        if eval:
            return np.random.rand() < self.epsilon_eval
        else:
            return self.steps < self.start_steps \
                   or np.random.rand() < self.epsilon_train.get()

    # def update_target(self):
    #     self.target_net = self.online_net.copy()

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

        mean_return = np.round(total_return / num_episodes, 1)

        print('-' * 60)
        print('Eval mean_steps: ', int(num_steps / num_episodes),
              'reward: ', mean_return)
        print('-' * 60)

        self.writer.add_scalar(
            'return/eval', mean_return, 4 * self.steps)
        # print(self.online_net.copy().reshape(self.env.nrow, self.env.ncol, 4)[0][1])

    def plot(self, q_value, dist=None):
        os.makedirs(self.log_dir + '/' + str(self.episodes))

        state_size = 3
        q_nrow = self.env.nrow * state_size
        q_ncol = self.env.ncol * state_size

        # Normalization
        xmin = -1
        xmax = 1
        q_value = (q_value - q_value.min()) / (q_value.max() - q_value.min()) * \
                  (xmax - xmin) + xmin

        # Delete Corner
        q_value[:, 0] = 0
        q_value[-1, :, 1] = 0
        q_value[:, -1, 2] = 0
        # q_value[0, :, 3] = 0

        value = np.zeros((q_nrow, q_ncol))

        # 0.Left, 1.Down, 2.Right, 3.Up, 4.Center
        value[1::3, ::3] += q_value[:, :, 0]
        value[2::3, 1::3] += q_value[:, :, 1]
        value[1::3, 2::3] += q_value[:, :, 2]
        # value[::3, 1::3] += q_value[:, :, 3]
        value[1::3, 1::3] += q_value.mean(axis=2)

        # Heatmap Plot
        fig = plt.figure(figsize=(6, 12))
        ax = fig.add_subplot(1, 1, 1)
        mappable0 = plt.imshow(value, cmap=cm.jet, interpolation="bilinear",
                               vmax=abs(value).max(), vmin=-abs(value).max())

        ax.set_xticks(np.arange(-0.5, q_ncol, 3))
        ax.set_yticks(np.arange(-0.5, q_nrow, 3))
        ax.set_xticklabels(range(self.env.ncol + 1))
        ax.set_yticklabels(range(self.env.nrow + 1))
        ax.grid(which="both")

        # Marker Of Start, Goal, Cliff
        # Start: green, Goal: blue, Cliff: red
        for i in range(0, self.env.nrow):
            y = i * 3 + 1
            for j in range(self.env.ncol):
                x = j * 3 + 1
                if self.env.desc[i][j] == b'S':
                    ax.plot([x], [y], marker="o", color='g', markersize=40, alpha=0.8)
                    ax.text(x, y + 0.3, 'START', ha='center', size=12, c='w')
                elif self.env.desc[i][j] == b'G':
                    ax.plot([x], [y], marker="o", color='r', markersize=40, alpha=0.8)
                    ax.text(x, y + 0.3, 'GOAL', ha='center', size=12, c='w')
                elif self.env.desc[i][j] == b'H':
                    ax.plot([x], [y], marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
                elif self.env.desc[i][j] == b'g':
                    ax.plot([x], [y], marker="o", color='orange', markersize=30, markeredgewidth=10, alpha=0.8)

        fig.colorbar(mappable0, ax=ax, orientation="vertical")

        plt.savefig(
            self.log_dir + '/' +
            str(self.episodes) + '/' +
            'heatmap.png')
        plt.close()


        # Optimization Path
        fig = plt.figure(figsize=(6, 12))
        ax = fig.add_subplot(1, 1, 1)
        value = np.zeros((q_nrow, q_ncol))
        mappable0 = plt.imshow(value, cmap=cm.jet, interpolation="bilinear",
                               vmax=abs(value).max(), vmin=-abs(value).max())

        opt_act = q_value.argmax(axis=2)
        self.plot_arrow(ax, opt_act)

        ax.set_xticks(np.arange(-0.5, q_ncol, 3))
        ax.set_yticks(np.arange(-0.5, q_nrow, 3))
        ax.set_xticklabels(range(self.env.ncol + 1))
        ax.set_yticklabels(range(self.env.nrow + 1))
        ax.grid(which="both")

        # Marker Of Start, Goal, Cliff
        # Start: green, Goal: blue, Cliff: red
        for i in range(0, self.env.nrow):
            y = i * 3 + 1
            for j in range(self.env.ncol):
                x = j * 3 + 1
                if self.env.desc[i][j] == b'S':
                    ax.plot([x], [y], marker="o", color='g', markersize=40, alpha=0.8)
                    ax.text(x, y + 0.3, 'START', ha='center', size=12, c='w')
                elif self.env.desc[i][j] == b'G':
                    ax.plot([x], [y], marker="o", color='r', markersize=40, alpha=0.8)
                    ax.text(x, y + 0.3, 'GOAL', ha='center', size=12, c='w')
                elif self.env.desc[i][j] == b'H':
                    ax.plot([x], [y], marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
                elif self.env.desc[i][j] == b'g':
                    ax.plot([x], [y], marker="o", color='orange', markersize=30, markeredgewidth=10, alpha=0.8)

        fig.colorbar(mappable0, ax=ax, orientation="vertical")

        plt.savefig(
            self.log_dir + '/' +
            str(self.episodes) + '/' +
            'optimization.png')
        plt.close()

        # Distribution Plot
        if dist is not None:
            # sns.set(rc={"figure.figsize": (6, 12)});
            plt.gcf().set_size_inches(6, 12)

            for i in range(self.env.nrow * self.env.ncol):
                subplot(self.env.nrow, self.env.ncol, i + 1,
                        facecolor='#EAEAF2', fc='#EAEAF2')
                for x in ['left', 'bottom', 'top', 'right']:
                    plt.gca().spines[x].set_visible(False)
                    # plt.gca().spines['top'].set_visible(False)

                # for j, c in zip(range(4), ['red', 'blue', 'green', 'darkorange']):
                for j, c in enumerate(['red', 'blue', 'green']):
                    ax = sns.distplot(dist[i, :, j], color=c, hist=False)
                    ax.fill_between(ax.lines[j].get_xydata()[:, 0],
                                    ax.lines[j].get_xydata()[:, 1],
                                    color=c, alpha=0.3)

            # # one liner to remove *all axes in all subplots*
            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);

            plt.savefig(
                self.log_dir + '/' +
                str(self.episodes) + '/' +
                'distribution.png')
            plt.close()

    def plot_arrow(self, ax, opt_act):
        for y in range(self.env.nrow):
            for x in range(1, self.env.ncol):
                if opt_act[y][x] == 0:   # 右向き
                    ax.annotate('', xy=[0+3*x, 1+3*y], xytext=[2+3*x, 1+3*y],
                        arrowprops=dict(shrink=10, width=20, headwidth=40,
                        headlength=20, connectionstyle='arc3',
                        facecolor='red', edgecolor='red'))
                elif opt_act[y][x] == 1:   # 下向き
                    ax.annotate('', xy=[1+3*x, 2+3*y], xytext=[1+3*x, 0+3*y],
                        arrowprops=dict(shrink=10, width=20, headwidth=40,
                        headlength=20, connectionstyle='arc3',
                        facecolor='red', edgecolor='red'))
                elif opt_act[y][x] == 2:   # 左向き
                    ax.annotate('', xy=[2+3*x, 1+3*y], xytext=[0+3*x, 1+3*y],
                        arrowprops=dict(shrink=10, width=20, headwidth=40,
                        headlength=20, connectionstyle='arc3',
                        facecolor='red', edgecolor='red'))

    def __del__(self):
        self.env.close()
        self.test_env.close()
        self.writer.close()
