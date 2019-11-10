import os
import numpy as np
import time
from common.env import make_pytorch_env
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque

class Abstract:
    def __init__(self, args):
        # param
        self.gamma = args.gamma
        self.priority_exp = args.priority_exp
        self.multi_step = args.multi_step
        self.save_model_interval = args.save_model_interval
        self.save_log_interval = args.save_log_interval
        self.batch_size = args.batch_size
        self.update_per_epoch = args.update_per_epoch

        # interval process
        self.eval_interval = args.learner_eval
        self.target_update_interval = args.learner_target_update
        self.save_log_interval = args.learner_save_log

        # env
        self.env = make_pytorch_env(args.env_id)
        self.n_state = self.env.observation_space.shape[0]
        self.n_act = self.env.action_space.n

        # logs
        self.summary_path = os.path.join('./', 'logs', 'summary')
        self.model_path = os.path.join('./', 'logs', 'model')
        self.writer = SummaryWriter(log_dir=self.summary_path)

        # device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def add_episode_buff(self, state, action, done):
        state = state[-int(self.n_state/4):].copy()

        state = np.array(state * 255, dtype=np.uint8)

        self.episode_buff['state'].append(state)
        self.episode_buff['action'].append(action)
        self.episode_buff['done'].append(done)

    def calc_discount_reward(self):
        multi_reward = list(self.reward_deque)
        discount_reward = 0
        for i, r in enumerate(multi_reward):
            discount_reward += (r * (self.gamma ** i))
        return discount_reward

    def process_batch(self, batch):
        batch['state'] = torch.FloatTensor(batch['state'] / 255.0).to(self.device)
        batch['action'] = torch.LongTensor(batch['action']).to(self.device)
        batch['reward'] = torch.FloatTensor(batch['reward']).to(self.device)
        batch['done'] = torch.FloatTensor(batch['done']).to(self.device)
        batch['next_state'] = torch.FloatTensor(batch['next_state'] / 255.0).to(self.device)
        return batch

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

    def interval(self):
        if self.epochs % self.eval_interval == 0:
            self.evaluate()

        if self.epochs % self.save_model_interval*100 == 0:
            self.net.save(self.model_path)
            self.target_net.save(self.model_path, target=True)

        if self.epochs % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.net.state_dict())
            self.target_net.eval()

        self.train_log()

    def agent_reset(self):
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
            'reward/test', mean_return, self.n_train)
        print('Learer  '
              f'Num steps: {self.n_train:<5} '
              f'reward: {mean_return:<5.1f}+/- {std_return:<5.1f}')

    def exploit(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.net(state).argmax().item()
        return action

    def train_log(self):
        now = time.time()
        if self.epochs % self.save_log_interval == 0:
            self.writer.add_scalar(
                "loss/learner",
                self.total_loss,
                self.epochs)
        print(
            f"Learer: loss: {self.total_loss:< 8.3f} "
            f"memory: {len(self.memory):<5} \t"
            f"time: {now - self.time:.2f}s")
        self.time = now

    def agent_log(self):
        now = time.time()
        if self.n_steps % self.save_log_interval == 0:
            self.writer.add_scalar(
                    f"mean_reward",
                    self.reward_logger.mean_reward(),
                    self.n_episodes)
        print(
            " "*20,
            f"episode {self.n_episodes:<5} \t"
            f"step {self.n_steps:<5} \t"
            f"reward {self.episode_reward:< 7.3f} \t"
            f"time: {now - self.time:.2f}s")
        self.time = now
