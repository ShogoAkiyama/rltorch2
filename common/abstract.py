import os
import time
from collections import deque
import numpy as np
from copy import deepcopy
import torch

from common.env import make_pytorch_env


class Abstract:
    def __init__(self, args, shared_memory, shared_weights):
        self.time = time.time()
        self.recurrent = args.recurrent
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # path
        self.memory_path = os.path.join(
            './', 'logs', 'memory')
        self.model_path = os.path.join(
            './', 'logs', 'model')

        self.env = make_pytorch_env(args.env_id)
        self.n_state = self.env.observation_space.shape[0]
        self.n_act = self.env.action_space.n

        # agent_log
        self.save_memory_interval = args.actor_save_memory
        self.load_model_interval = args.actor_load_model
        self.actor_save_log_interval = args.actor_save_log

        # learner_log
        self.eval_interval = args.learner_eval
        self.load_memory_interval = args.learner_load_memory
        self.save_model_interval = args.learner_save_model
        self.target_update_interval = args.learner_target_update
        self.learner_save_log_interval = args.learner_save_log

        # param
        self.gamma = args.gamma
        self.priority_exp = args.priority_exp
        self.multi_step = args.multi_step

        # learner param
        self.batch_size = args.batch_size
        self.optim_lr = args.lr
        self.update_per_epoch = args.update_per_epoch

        self.key_list = ['state', 'action', 'reward', 'done', 'priority']

        # share
        self.shared_memory = shared_memory
        self.shared_weights = shared_weights

    def actor_run(self):
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

                self.agent_reset()

    def learner_run(self):
        while len(self.memory) <= 1:#self.batch_size:
            while not self.shared_memory.empty():
                batch = self.shared_memory.get()
                self.memory.load(batch)

            time.sleep(1.0)

        self.time = time.time()
        while True:
            self.epochs += 1
            self.train()
            self.learner_interval()

    def interact(self):
        state = self.env_state
        action = self.select_action(state)
        done = self.env_done
        next_state, reward, next_done, _ = self.env.step(action)

        self.episode_reward += reward

        self.add_episode_buff(state, action, done)

        if next_done:
            self.actor_log()
            self.n_episodes += 1
            self.episode_done = True

            self.reward_deque.append(reward)

            while len(self.reward_deque) > 0:
                if self.recurrent:
                    self.episode_buff['recc'].append(np.zeros((2, 512)))
                    self.episode_buff['target_recc'].append(np.zeros((2, 512)))

                self.episode_buff['state'].append(
                    np.zeros((int(self.n_state/4), 84, 84), dtype=np.uint8))

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
        if self.recurrent:
            batch['state'] = batch['state'][:, self.mask].view(self.batch_size,
                                                               self.seq_size + self.multi_step,
                                                               self.n_state, 84, 84)
            batch['recc'] = torch.FloatTensor(batch['recc']).to(self.device)
            batch['target_recc'] = torch.FloatTensor(batch['target_recc']).to(self.device)
        else:
            batch['next_state'] = torch.FloatTensor(batch['next_state'] / 255.0).to(self.device)
        return batch

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
            if self.recurrent:
                self.net.reset_recc()
                self.target_net.reset_recc()
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
            'reward/test', mean_return, self.train_steps)
        print('Learer  '
              f'Num steps: {self.train_steps:<5} '
              f'reward: {mean_return:<5.1f}+/- {std_return:<5.1f}')

    def learner_interval(self):
        if self.epochs % self.eval_interval == 0:
            self.evaluate()

        if self.epochs % self.load_memory_interval == 0:
            while not self.shared_memory.empty():
                batch = self.shared_memory.get()
                self.memory.load(batch)
            # self.memory.load()

        if self.epochs % self.save_model_interval == 0:
            self.save_model()

        if self.epochs % self.save_model_interval*100 == 0:
            self.net.save(self.model_path)
            self.target_net.save(self.model_path, target=True)

        if self.epochs % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.net.state_dict())
            self.target_net.eval()

        self.learner_log()

    def exploit(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.net(state).argmax().item()
        return action

    def load_model(self):
        try:
            self.net.load_state_dict(self.shared_weights['net_state'])
            self.target_net.load_state_dict(self.shared_weights['target_net_state'])
        except:
            print('load error')

    def save_model(self):
        self.shared_weights['net_state'] = deepcopy(self.net).cpu().state_dict()
        self.shared_weights['target_net_state'] = deepcopy(self.target_net).cpu().state_dict()

    def actor_log(self):
        now = time.time()
        if self.n_steps % self.actor_save_log_interval == 0:
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

    def learner_log(self):
        now = time.time()
        if self.epochs % self.learner_save_log_interval == 0:
            self.writer.add_scalar(
                "loss/learner",
                self.total_loss,
                self.epochs)
        print(
            f"Learer: loss: {self.total_loss:< 8.3f} "
            f"memory: {len(self.memory):<5} \t"
            f"time: {now - self.time:.2f}s")
        self.time = now
