import torch
import numpy as np
from network import Network
import torch.optim as optim
import math
import gym

from utils.ReplayMemory import ExperienceReplayMemory, PrioritizedReplayMemory


class QRDQN:
    def __init__(self, args=None):
        self.num_quantiles = args.QUANTILES
        self.cumulative_density = torch.tensor((2 * np.arange(self.num_quantiles) + 1) / (2.0 * self.num_quantiles),
                                               device=args.device, dtype=torch.float)
        self.quantile_weight = 1.0 / self.num_quantiles
        self.max_frames = args.max_frames

        # DQN
        self.device = args.device

        self.env = gym.make(args.env_id)
        self.env_eval = gym.make(args.env_id)

        self.num_feats = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        # epsilon
        self.epsilon_start = args.epsilon_start
        self.epsilon_final = args.epsilon_final
        self.epsilon_decay = args.epsilon_decay

        self.gamma = args.gamma
        self.lr = args.learning_rate
        self.target_net_update_freq = args.target_net_update_freq
        self.experience_replay_size = args.exp_replay_size
        self.batch_size = args.batch_size
        self.learn_start = args.learn_start
        self.update_freq = args.update_freq

        self.declare_networks()

        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        self.model.train()
        self.target_model.train()

        self.update_count = 0

        self.memory = ExperienceReplayMemory(self.experience_replay_size)

        self.nsteps = args.n_steps
        self.nstep_buffer = []

        # base
        self.rewards = []

    def train_episode(self):
        episode_reward = 0

        state = self.env.reset()
        for frame_idx in range(1, self.max_frames + 1):
            epsilon = self.epsilon_by_frame(frame_idx)

            action = self.get_action(state, epsilon)
            prev_state = state
            state, reward, done, _ = self.env.step(action)
            state = None if done else state

            self.update(prev_state, action, reward, state, frame_idx)
            episode_reward += reward

            if done:
                self.finish_nstep()
                state = self.env.reset()
                self.rewards.append(reward)
                print('episode_reward: ', episode_reward)
                episode_reward = 0
                if frame_idx % 10:
                    rewards = self.evaluation(self.env_eval)
                    rewards_mu = np.array([np.sum(np.array(l_i), 0) for l_i in rewards]).mean()
                    print(" " * 20, "Eval Average Reward %.2f" % (rewards_mu))

    def declare_networks(self):
        self.model = Network(self.num_feats, self.num_actions, quantiles=self.num_quantiles)
        self.target_model = Network(self.num_feats, self.num_actions, quantiles=self.num_quantiles)

    def next_distribution(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        with torch.no_grad():
            quantiles_next = torch.zeros((self.batch_size, self.num_quantiles), device=self.device, dtype=torch.float)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                quantiles_next[non_final_mask] = self.target_model(non_final_next_states).gather(1,max_next_action).squeeze(dim=1)

            quantiles_next = batch_reward + (self.gamma * quantiles_next)

        return quantiles_next

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        batch_action = batch_action.unsqueeze(dim=-1).expand(-1, -1, self.num_quantiles)

        quantiles = self.model(batch_state)
        quantiles = quantiles.gather(1, batch_action).squeeze(1)

        quantiles_next = self.next_distribution(batch_vars)

        diff = quantiles_next.t().unsqueeze(-1) - quantiles.unsqueeze(0)

        loss = self.huber(diff) * torch.abs(self.cumulative_density.view(1, -1) - (diff < 0).to(torch.float))
        loss = loss.transpose(0, 1)
        loss = loss.mean(1).sum(-1).mean()

        return loss

    def get_action(self, s, eps):
        with torch.no_grad():
            if np.random.random() >= eps:
                action = self.action(s)
                return action
            else:
                action = np.random.randint(0, self.num_actions)
                return action

    def action(self, s):
        X = torch.tensor([s], device=self.device, dtype=torch.float)
        a = (self.model(X) * self.quantile_weight).sum(dim=2).max(dim=1)[1]
        return a.item()

    def get_max_next_state_action(self, next_states):
        next_dist = self.target_model(next_states) * self.quantile_weight
        return next_dist.sum(dim=2).max(1)[1].view(next_states.size(0), 1, 1).expand(-1, -1, self.num_quantiles)

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)

    def finish_nstep(self):
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2] * (self.gamma ** i) for i in range(len(self.nstep_buffer))])
            state, action, _, _ = self.nstep_buffer.pop(0)

            self.memory.push((state, action, R, None))

    def update(self, s, a, r, s_, frame=0):
        self.append_to_replay(s, a, r, s_)

        if frame < self.learn_start or frame % self.update_freq != 0:
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()

    def update_target_model(self):
        self.update_count += 1
        if self.update_count % self.target_net_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def append_to_replay(self, s, a, r, s_):
        self.nstep_buffer.append((s, a, r, s_))

        if (len(self.nstep_buffer) < self.nsteps):
            return

        R = sum([self.nstep_buffer[i][2] * (self.gamma ** i) for i in range(self.nsteps)])
        state, action, _, _ = self.nstep_buffer.pop(0)

        self.memory.push((state, action, R, s_))

    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)

        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (-1,) + self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device,
                                      dtype=torch.bool)
        try:  # sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device,
                                                 dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights

    def append_to_replay(self, s, a, r, s_):
        self.nstep_buffer.append((s, a, r, s_))

        if (len(self.nstep_buffer) < self.nsteps):
            return

        R = sum([self.nstep_buffer[i][2] * (self.gamma ** i) for i in range(self.nsteps)])
        state, action, _, _ = self.nstep_buffer.pop(0)

        self.memory.push((state, action, R, s_))

    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)

        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (-1,) + self.num_feats

        batch_state = torch.FloatTensor(batch_state).to(self.device).view(shape)
        batch_action = torch.LongTensor(batch_action).to(self.device).squeeze().view(-1, 1)
        batch_reward = torch.FloatTensor(batch_reward).to(self.device).squeeze().view(-1, 1)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device,
                                      dtype=torch.bool)
        try:  # sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device,
                                                 dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights

    def evaluation(self, env_eval):
        rewards = []
        for i in range(10):
            rewards_i = []

            state = env_eval.reset()
            action = self.action(state)
            state, reward, done, _ = env_eval.step(action)
            rewards_i.append(reward)

            while not done:
                action = self.action(state)
                state, reward, done, _ = env_eval.step(action)
                rewards_i.append(reward)
            rewards.append(rewards_i)

        return rewards

    def epsilon_by_frame(self, frame_idx):
        res = self.epsilon_final + (self.epsilon_start - self.epsilon_final) \
              * math.exp(-1. * frame_idx / self.epsilon_decay)

        return res
