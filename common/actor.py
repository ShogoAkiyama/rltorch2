import os
import time
from collections import deque
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from common.env import make_pytorch_env

class AbstractActor:

    def __init__(self, actor_id, args):
        self.actor_id = actor_id
        self.time = time.time()

        # path
        self.memory_path = os.path.join(
            './', 'logs', 'memory')
        self.model_path = os.path.join(
            './', 'logs', 'model')
        self.summary_path = os.path.join(
            './', 'logs', 'summary',
            f'actor-{self.actor_id}')

        self.env = make_pytorch_env(args.env_id)
        self.n_state = self.env.observation_space.shape[0]
        self.n_act = self.env.action_space.n

        # log
        self.writer = SummaryWriter(log_dir=self.summary_path)
        self.save_memory_interval = args.actor_save_memory
        self.load_model_interval = args.actor_load_model
        self.save_log_interval = args.actor_save_log

        # param
        self.gamma = args.gamma
        self.eps_greedy = 0.4 ** (1 + actor_id * 7 / (args.n_actors - 1)) \
            if args.n_actors > 1 else 0.4
        self.priority_exp = args.priority_exp
        self.multi_step = args.multi_step

        self.interval()

        # env
        self.n_episodes = 0

    def run(self):
        self.time = time.time()
        while True:
            self.n_steps += 1

            self.interact()

            if self.episode_done:
                if self.n_steps >= self.multi_step:
                    self.memory.add(self.episode_buff)
                    if self.n_episodes % self.save_memory_interval == 0:
                        self.shared_memory.put(self.memory.get())
                        self.memory.reset()

                if self.n_episodes % self.load_model_interval == 0:
                    self.load_model()

                # reset arena
                self.reward_logger.add_reward(self.episode_reward)
                self.env_state = self.env.reset()

                self.interval()

    def load_model(self):
        try:
            self.net.load_state_dict(self.shared_weights['net_state'])
            self.target_net.load_state_dict(self.shared_weights['target_net_state'])
        except:
            print('load error')

    def interval(self):
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

    def log(self):
        now = time.time()
        if self.n_steps % self.save_log_interval == 0:
            self.writer.add_scalar(
                    f"mean_reward",
                    self.reward_logger.mean_reward(),
                    self.n_episodes)
        print(
            " "*20,
            f"Actor {self.actor_id:<2}: \t"
            f"episode {self.n_episodes:<5} \t"
            f"step {self.n_steps:<5} \t"
            f"reward {self.episode_reward:< 7.3f} \t"
            f"time: {now - self.time:.2f}s")
        self.time = now

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
