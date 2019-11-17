import os
import torch
import numpy as np
from collections import deque

from .replay_memory import R2D2ReplayMemory

from .model import LSTMNetwork
from common.utils import grad_false
from common.logger import RewardLogger
from common.abstract import Abstract

from torch.utils.tensorboard import SummaryWriter


def actor_process(args, actor_id, shared_memory, shared_weights):
    actor = Actor(args, actor_id, shared_memory, shared_weights)
    actor.actor_run()


class Actor(Abstract):
    def __init__(self, args, actor_id, shared_memory, shared_weights):
        super(Actor, self).__init__(args, shared_memory, shared_weights)
        self.actor_id = actor_id

        # r2d2 param
        self.seq_size = args.seq_size
        self.overlap_size = args.overlap_size
        self.eta = args.eta
        self.overlap = None

        # reward logger
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
        add_key = ['recc', 'target_recc']
        self.shared_memory = shared_memory
        self.memory = R2D2ReplayMemory(args, add_key)

        # network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = LSTMNetwork(self.n_state, self.n_act).eval().apply(grad_false)
        self.target_net = LSTMNetwork(self.n_state, self.n_act).eval().apply(grad_false)
        self.net.to(self.device)
        self.target_net.to(self.device)

        self.load_model()
        self.env_state = self.env.reset()

        self.key_list += add_key
        self.agent_reset()

    def agent_reset(self):
        super().agent_reset()
        self.net.reset_recc()
        self.target_net.reset_recc()
        self.td_loss = deque(maxlen=self.seq_size)
        self.overlap = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        with torch.no_grad():
            curr_q = self.net(state).cpu().numpy().reshape(-1)
            action = int(curr_q.argmax())
            self.next_q = self.target_net(state).cpu().numpy().reshape(-1)[action]

        recc = self.net.get_recc().cpu().numpy().reshape(2, -1)
        target_recc = self.target_net.get_recc().cpu().numpy().reshape(2, -1)
        self.episode_buff['recc'].append(recc)
        self.episode_buff['target_recc'].append(target_recc)

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

        td_loss = np.abs(target_q - curr_q)
        self.td_loss.append(td_loss)

        self.overlap += 1

        # add priority
        if (len(self.td_loss) == self.seq_size) and \
                ((self.overlap >= self.overlap_size) or done):
            priority = self.eta * np.max(self.td_loss) +\
                       (1. - self.eta) * np.mean(self.td_loss)
            priority = np.power(priority, self.priority_exp)
            self.episode_buff['priority'].append(priority)
            self.overlap = 0
