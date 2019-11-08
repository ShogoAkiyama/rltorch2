import torch
import numpy as np
from collections import deque
import time

from .replay_memory import R2D2ReplayMemory

from .model import LSTMNetwork
from common.utils import grad_false
from common.logger import RewardLogger
from common.actor import AbstractActor

def actor_process(actor_id, args, shared_memory, shared_weights):
    actor = Actor(actor_id, args, shared_memory, shared_weights)
    actor.run()


class Actor(AbstractActor):

    def __init__(self, actor_id, args, shared_memory, shared_weights):
        super(Actor, self).__init__(actor_id, args)
        self.seq_size = args.seq_size
        self.overlap_size = args.overlap_size
        self.eta = args.eta

        # reward logger
        self.reward_logger = RewardLogger()

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

        self.shared_weights = shared_weights
        self.load_model()
        self.env_state = self.env.reset()

        self.key_list = ['state', 'action', 'reward', 'done', 'priority'] + add_key
        self.inerval()

    def inerval(self):
        super().interval()
        self.net.reset_recc()
        self.target_net.reset_recc()
        self.td_loss = deque(maxlen=self.seq_size)
        self.overlap = 0

    def run(self):
        self.time = time.time()
        while True:
            self.n_steps += 1

            self.interact()

            if self.episode_done:
                if self.n_steps >= self.seq_size + self.multi_step:
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
                self.episode_buff['recc'].append(np.zeros((2, 512)))
                self.episode_buff['target_recc'].append(np.zeros((2, 512)))

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


