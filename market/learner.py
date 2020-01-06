import torch
import numpy as np
from network import Network
import torch.optim as optim
import gym
import matplotlib.pyplot as plt
import seaborn as sns
from env import make_pytorch_env


class Learner:
    def __init__(self, args, q_batch):
        self.q_batch = q_batch
        self.num_quantiles = args.quantiles
        self.device = args.device

        self.cumulative_density = torch.FloatTensor((2 * np.arange(self.num_quantiles) + 1) / (2.0 * self.num_quantiles)).to(self.device)
        self.quantile_weight = 1.0 / self.num_quantiles

        self.device = args.device

        self.env_eval = make_pytorch_env()

        self.num_feats = self.env_eval.observation_space
        self.num_actions = self.env_eval.action_space

        # epsilon
        self.gamma = args.gamma
        self.lr = args.learning_rate
        self.target_net_update_freq = args.target_net_update_freq

        self.batch_size = args.batch_size
        self.learn_start = args.learn_start
        self.update_freq = args.update_freq

        self.model = Network(self.num_feats, self.num_actions, quantiles=self.num_quantiles).to(self.device)
        self.target_model = Network(self.num_feats, self.num_actions, quantiles=self.num_quantiles).to(self.device)

        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.update_count = 0

    def learn(self):
        while True:
            self.update_count += 1
            if self.update_count % self.target_net_update_freq == 0:
                self.update_target_model()

            if self.update_count % 10 == 0:
                rewards = self.evaluation()
                rewards_mu = np.array([np.sum(np.array(l_i), 0) for l_i in rewards]).mean()
                print('Eval Reward %.2f' % (rewards_mu))

            states, actions, rewards, next_states, dones = self.q_batch.get(block=True)

            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device).view(-1, 1)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor([int(i) for i in dones]).to(self.device).view(-1, 1)

            actions = actions.unsqueeze(dim=-1).unsqueeze(dim=-1).expand(-1, -1, self.num_quantiles)

            curr_q = self.model(states).gather(1, actions).squeeze(1)

            with torch.no_grad():
                next_dist = self.target_model(next_states) * self.quantile_weight
                next_action = next_dist.sum(dim=2).max(1)[1].view(next_states.size(0), 1, 1).expand(
                    -1, -1, self.num_quantiles)

                next_q = self.target_model(next_states).gather(1, next_action).squeeze(dim=1)

                target_q = rewards + (self.gamma * next_q) * (1 - dones)

            diff = target_q.t().unsqueeze(-1) - curr_q.unsqueeze(0)

            loss = self.huber(diff) * torch.abs(self.cumulative_density.view(1, -1) - (diff < 0).to(torch.float))
            loss = loss.transpose(0, 1)
            loss = loss.mean(1).sum(-1).mean()

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)

    def update_target_model(self):
            self.target_model.load_state_dict(self.model.state_dict())

    def prep_minibatch(self):
        batch_state, batch_action, batch_reward, batch_next_state = self.q_batch.get(block=True)

        shape = (-1,) + self.num_feats

        batch_state = torch.FloatTensor(batch_state).to(self.device).view(shape)
        batch_action = torch.LongTensor(batch_action).to(self.device).squeeze().view(-1, 1)
        batch_reward = torch.FloatTensor(batch_reward).to(self.device).squeeze().view(-1, 1)

        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None, batch_next_state)
                                                )).to(self.device)
        try:  # sometimes all next states are false
            non_final_next_states = torch.FloatTensor([s for s in batch_next_state if s is not None]
                                                      ).to(self.device).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values

    def evaluation(self):
        rewards = []
        for i in range(10):
            rewards_i = []

            state = self.env_eval.reset()
            action = self.action(state)
            state, reward, done, _ = self.env_eval.step(action)
            rewards_i.append(reward)

            while not done:
                action = self.action(state)
                state, reward, done, _ = self.env_eval.step(action)
                rewards_i.append(reward)
            rewards.append(rewards_i)

        return rewards

    def action(self, state):
        state = torch.FloatTensor([state]).to(self.device)
        action = (self.model(state) * self.quantile_weight)
        if self.update_count > 5000:
            dist_action = action[0].cpu().detach().numpy()
            sns.distplot(dist_action[0], bins=10, color='red')
            sns.distplot(dist_action[1], bins=10, color='blue')
            plt.show()
        action = action.sum(dim=2).max(dim=1)[1]
        return action.item()
