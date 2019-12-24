import numpy as np
import gym
import torch
import torch.optim as optim
import torch.nn.functional as F

from model import ConvNet

class Learner:
    def __init__(self, args, q_batch):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_batch = q_batch
        self.learn_step_counter = 0
        self.gamma = args.gamma
        self.batch_size = args.batch_size

        self.env = gym.make(args.env)
        self.n_act = self.env.action_space.n
        self.n_state = self.env.observation_space.shape[0]
        self.n_quant = args.quant

        self.net = ConvNet(self.n_state, self.n_act, self.n_quant).to(self.device)
        self.target_net = ConvNet(self.n_state, self.n_act, self.n_quant).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr)

    def learn(self):
        while True:
            self.learn_step_counter += 1

            # target parameter update
            if self.learn_step_counter % 10 == 0:
                self.update_target()

            states, actions, rewards, next_states, dones = self.q_batch.get(block=True)
    
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = np.array([int(i) for i in dones])

            # action value distribution prediction
            # (m, N_ACTIONS, N_ATOM)
            curr_q, curr_q_tau = self.net(states)

            # 実際に行動したQだけを取り出す
            curr_q = torch.stack([curr_q[i].index_select(0, actions[i]) 
                                    for i in range(self.batch_size)]).squeeze(1)
            # curr_q = curr_q.unsqueeze(2)

            # get next state value
            next_q, _ = self.net(next_states)  # (m, N_ACTIONS, N_ATOM)
            next_action = next_q.mean(dim=2).argmax(dim=1)  # (m)

            # target_q
            # q_target = R + gamma * (1 - terminate) * q_next
            target_q, _ = self.target_net(next_states)
            target_q = target_q.detach().cpu().numpy()
            target_q = np.array([target_q[i, action, :] for i, action in enumerate(next_action)])
            target_q = rewards.reshape(-1, 1) + self.gamma * target_q * (1 - dones.reshape(-1, 1))
            target_q = torch.FloatTensor(target_q).to(self.device)

            # loss
            # u = target_q - curr_q  # (m, N_QUANT, N_QUANT)
            # tau = curr_q_tau.unsqueeze(0)  # (1, N_QUANT, 1)
            # weight = torch.abs(tau - u.le(0.).float())  # (m, N_QUANT, N_QUANT)

            loss = F.smooth_l1_loss(curr_q, target_q.detach(), reduction='none')

            # (m, N_QUANT, N_QUANT)
            loss = torch.mean(torch.sum(loss, dim=1))

            if self.learn_step_counter % 100 == 0:
                print('loss:', loss.item())

            # backprop loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())
