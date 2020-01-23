import numpy as np
import gym
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.optim as optim
import torch.nn.functional as F

from model import ConvNet

class Learner:
    def __init__(self, args, q_batch):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_batch = q_batch
        self.update_count = 0
        self.gamma = args.gamma
        self.batch_size = args.batch_size

        self.env_eval = gym.make(args.env)
        self.n_act = self.env_eval.action_space.n
        self.n_state = self.env_eval.observation_space.shape[0]
        self.n_quant = args.quant
        
        self.target_net_update_freq = args.target_net_update_freq

        self.net = ConvNet(self.n_state, self.n_act, self.n_quant).to(self.device)
        self.target_net = ConvNet(self.n_state, self.n_act, self.n_quant).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr)

    def learn(self):
        while True:
            self.update_count += 1

            if self.update_count % 10 == 0:
                rewards = self.evaluation()
                rewards_mu = np.array([np.sum(np.array(l_i), 0) 
                                       for l_i in rewards]).mean()
                print('update cnt %d Eval Reward %.2f' % (self.update_count, rewards_mu))

            # target parameter update
            if self.update_count % self.target_net_update_freq  == 0:
                self.update_target()

            states, actions, rewards, next_states, dones = self.q_batch.get(block=True)
    
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = np.array([int(i) for i in dones])

            # action value distribution prediction
            # [BATCH, N_QUANT, N_ACTIONS]
            curr_q, _ = self.net(states)

            # 実際に行動したQだけを取り出す
            # [BATCH, N_QUANT, 1]
            curr_q = torch.stack([curr_q[i].index_select(1, actions[i])
                                    for i in range(self.batch_size)])
            
            # # [BATCH, N_QUANT, N_QUANT]
            curr_q = curr_q.repeat(1, 1, self.n_quant)
            curr_q = curr_q.permute(0, 2, 1)

            # get next state value
            # [BATCH, N_QUANT, N_ACTIONS]
            next_q, _ = self.net(next_states)
            next_action = next_q.sum(dim=1).argmax(dim=1)

            # target_q

            # [BATCH, N_QUANT, N_ACT]
            target_q, _ = self.target_net(next_states)
            target_q = target_q.detach().cpu().numpy()

            # [BATCH, N_QUANT, 1]
            target_q = np.array([target_q[i, :, action] 
                                 for i, action in enumerate(next_action)])
            target_q = rewards.reshape(-1, 1) + self.gamma * target_q * (1 - dones.reshape(-1, 1))
            target_q = torch.FloatTensor(target_q).to(self.device).unsqueeze(2)

            # # [BATCH, N_QUANT, N_QUANT]
            target_q = target_q.repeat(1, 1, self.n_quant)

            loss = F.smooth_l1_loss(curr_q, target_q.detach(), reduction='none')

            # (BATCH, N_QUANT, N_QUANT)
            # huber loss
            loss = torch.mean(torch.sum(torch.mean(loss, dim=2), dim=1))

            # backprop loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def evaluation(self):
        rewards = []
        for _ in range(10):
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
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
    
        action_value, _ = self.net(state)
        # if self.update_count > 3000:
        #     dist_action = action_value[0].detach().cpu().numpy()
        #     sns.distplot(dist_action[:, 0], bins=10, color='red')
        #     sns.distplot(dist_action[:, 1], bins=10, color='blue')
        #     plt.show()

        action_value = action_value[0].sum(dim=0)
        action = torch.argmax(action_value).detach().cpu().item()
        return action