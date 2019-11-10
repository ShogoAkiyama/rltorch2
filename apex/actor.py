import os

import torch
import numpy as np
import gym
import time

from common.replay_memory import EpisodeReplayMemory

from .model import QNetwork
from common.utils import grad_false
from common.logger import RewardLogger
from common.actor import AbstractActor

def actor_process(actor_id, args, shared_memory, shared_weights):
    actor = Actor(actor_id, args, shared_memory, shared_weights)
    actor.run()


class Actor(AbstractActor):

    def __init__(self, actor_id, args, shared_memory, shared_weights):
        super(Actor, self).__init__(actor_id, args)

        # reward logger
        self.reward_logger = RewardLogger()

        # memory
        self.shared_memory = shared_memory
        self.memory = EpisodeReplayMemory(args)

        # network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = QNetwork(self.n_state, self.n_act).eval().apply(grad_false)
        self.target_net = QNetwork(self.n_state, self.n_act).eval().apply(grad_false)

        self.net.to(self.device)
        self.target_net.to(self.device)

        self.shared_weights = shared_weights
        self.load_model()

        self.env_state = self.env.reset()

        self.key_list = ['state', 'action', 'reward', 'done', 'priority']

        self.interval()

    def interact(self):
        state = self.env_state
        action = self.select_action(state)
        done = self.env_done
        next_state, reward, next_done, _ = self.env.step(action)

        self.episode_reward += reward

        self.add_episode_buff(state, action, done)

        if next_done:
            self.log()
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


