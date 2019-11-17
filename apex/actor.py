import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from common.replay_memory import EpisodeReplayMemory

from .model import QNetwork
from common.utils import grad_false
from common.logger import RewardLogger
from common.abstract import Abstract


def actor_process(args, actor_id, shared_memory, shared_weights):
    actor = Actor(args, actor_id, shared_memory, shared_weights)
    actor.actor_run()


class Actor(Abstract):
    def __init__(self, args, actor_id, shared_memory, shared_weights):
        super(Actor, self).__init__(args, shared_memory, shared_weights)
        self.actor_id = actor_id

        self.summary_path = os.path.join(
            './', 'logs', 'summary', f'actor-{self.actor_id}')
        self.writer = SummaryWriter(log_dir=self.summary_path)
        self.reward_logger = RewardLogger()

        # param
        self.eps_greedy = 0.4 ** (1 + actor_id * 7 / (args.n_actors - 1)) \
            if args.n_actors > 1 else 0.4
        self.n_steps = 0
        self.n_episodes = 0

        # memory
        self.memory = EpisodeReplayMemory(args)

        # network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = QNetwork(self.n_state, self.n_act).eval().apply(grad_false)
        self.target_net = QNetwork(self.n_state, self.n_act).eval().apply(grad_false)
        self.net.to(self.device)
        self.target_net.to(self.device)
        self.load_model()

        self.env_state = self.env.reset()
        self.agent_reset()

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

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


