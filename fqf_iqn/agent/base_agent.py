import os
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from memory import LazyMultiStepMemory
from utils import RunningMeanStats, LinearAnneaer


class BaseAgent:

    def __init__(self, env, test_env, log_dir, num_steps=5*(10**7),
                 batch_size=32, memory_size=10**6, gamma=0.99, multi_step=1,
                 update_interval=4, target_update_interval=10000,
                 start_steps=50000, epsilon_train=0.01, epsilon_eval=0.001,
                 epsilon_decay_steps=250000, double_q_learning=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=27000, grad_cliping=5.0, cuda=True, seed=0):

        self.env = env
        self.test_env = test_env

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        self.test_env.seed(2**31-1-seed)

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.online_net = None
        self.target_net = None

        # Replay memory which is memory-efficient to store stacked frames.
        self.memory = LazyMultiStepMemory(
            memory_size, self.env.observation_space.n,
            self.device, gamma, multi_step)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_return = RunningMeanStats(log_interval)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.num_actions = self.env.action_space.n
        self.num_steps = num_steps
        self.batch_size = batch_size

        self.double_q_learning = double_q_learning

        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.num_eval_steps = num_eval_steps
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.epsilon_train = LinearAnneaer(
            1.0, epsilon_train, epsilon_decay_steps)
        self.epsilon_eval = epsilon_eval
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.max_episode_steps = max_episode_steps
        self.grad_cliping = grad_cliping

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
        self.target_net.load_state_dict(
            self.online_net.state_dict())

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
        self.online_net.train()
        self.target_net.train()

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

            self.memory.append(
                state, action, reward, next_state, done)

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

        if self.is_update():
            self.learn()

        if self.steps % self.eval_interval == 0:
            self.evaluate()
            self.save_models(os.path.join(self.model_dir, 'final'))

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

        print('-' * 60)
        print('Eval mean_steps: ', int(num_steps / num_episodes),
              'reward: ', np.round(total_return / num_episodes, 1))
        print('-' * 60)

        mean_return = total_return / num_episodes

        if mean_return > self.best_eval_score:
            self.best_eval_score = mean_return
            self.save_models(os.path.join(self.model_dir, 'best'))

        # We log evaluation results along with training frames = 4 * steps.
        self.writer.add_scalar(
            'return/test', mean_return, 4 * self.steps)

        self.plot()

    def __del__(self):
        self.env.close()
        self.test_env.close()
        self.writer.close()