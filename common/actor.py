import os
import time
from collections import deque

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
        self.episode_done = False
        self.next_q = None
        self.env_done = False
        self.episode_reward = 0
        self.n_steps = 0
        self.reward_deque = deque(maxlen=self.multi_step)
        self.curr_q_deque = deque(maxlen=self.multi_step)

        self.episode_buff = {}
        for key in ['state', 'action', 'reward', 'done', 'priority']:
            self.episode_buff[key] = list()

        # env
        self.n_episodes = 0

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
        for key in ['state', 'action', 'reward', 'done', 'priority']:
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