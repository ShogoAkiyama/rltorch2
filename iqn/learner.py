import numpy as np
import gym
import torch
import torch.optim as optim

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

        self.v_min = args.v_min
        self.v_max = args.v_max

        self.dz = float(self.v_max - self.v_min) / (self.n_quant - 1)
        # self.v_range = np.linspace(self.v_min, self.v_max, self.n_atom)
        # self.value_range = torch.FloatTensor(self.v_range).to(self.device)  # (N_ATOM)
        self.z = [self.v_min + i * self.dz for i in range(self.n_quant)]
        self.z_space = torch.FloatTensor(self.z).to(self.device)

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
            # dones = [int(i) for i in dones]

            # action value distribution prediction
            # (m, N_ACTIONS, N_ATOM)
            curr_q = self.net(states)

            # 実際に行動したQだけを取り出す
            curr_q = torch.stack([curr_q[i].index_select(0, actions[i]) 
                                    for i in range(self.batch_size)]).squeeze(1)

            # get next state value
            next_q = self.net(next_states).detach()  # (m, N_ACTIONS, N_ATOM)
            next_q = torch.sum(next_q * self.z_space.view(1, 1, -1), dim=2)  # (m, N_ACTIONS)
            next_action = next_q.argmax(dim=1)  # (m)

            # target_q
            target_q = self.target_net(next_states).detach().cpu().numpy()
            target_q = [target_q[i, action, :] for i, action in enumerate(next_action)]
            target_q = np.array(target_q)  # (m, N_ATOM)

            # Tz = rewards.reshape(-1, 1) + (1 - dones.reshape(-1, 1)) * 0.99 * np.expand_dims(self.z_space.data.cpu().numpy(),0)
            # Tz = np.clip(Tz, self.v_min, self.v_max)

            m_prob = np.zeros((self.batch_size, self.n_atom))  # (m, N_ATOM)

            # we didn't vectorize the computation of target assignment.
            for i in range(self.batch_size):
                for j in range(self.n_atom):
                    Tz = np.fmin(self.v_max,
                            np.fmax(self.v_min,
                                    rewards[i]
                                    + (1 - dones[i]) * 0.99 * (self.v_min + j * self.dz)
                                    )
                            )

                    bj = (Tz - self.v_min) / self.dz

                    lj = np.floor(bj).astype(int)   # m_l
                    uj = np.ceil(bj).astype(int)    # m_u

                    # calc prob mass of relative position weighted with distance
                    m_prob[i, lj] += (dones[i] + (1 - dones[i]) * target_q[i][j]) * (uj - bj)
                    m_prob[i, uj] += (dones[i] + (1 - dones[i]) * target_q[i][j]) * (bj - lj)

            m_prob = m_prob / m_prob.sum(axis=1, keepdims=1)

            m_prob = torch.FloatTensor(m_prob).to(self.device)
            # print(curr_q)

            # calc huber loss, dont reduce for importance weight
            loss = - torch.mean(torch.sum(m_prob * torch.log(curr_q + 1e-20), dim=1))  # (m , N_ATOM)
            
            if self.learn_step_counter % 100 == 0:
                print('loss:', loss.item())

            # backprop loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())
