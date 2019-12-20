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

        self.env = gym.make(args.env)
        self.n_act = self.env.action_space.n
        self.n_state = self.env.observation_space.shape[0]
        self.n_atom = args.atom

        self.v_min = args.v_min
        self.v_max = args.v_max

        self.v_step = ((self.v_max - self.v_min) / (self.n_atom - 1))
        self.v_range = np.linspace(self.v_min, self.v_max, self.n_atom)
        self.value_range = torch.FloatTensor(self.v_range).to(self.device)  # (N_ATOM)

        self.pred_net = ConvNet(self.n_state, self.n_act, self.n_atom).to(self.device)
        self.target_net = ConvNet(self.n_state, self.n_act, self.n_atom).to(self.device)
        self.optimizer = optim.Adam(self.pred_net.parameters(), lr=args.lr)

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

            # action value distribution prediction
            # (m, N_ACTIONS, N_ATOM)
            curr_q = self.pred_net(states)
            mb_size = curr_q.size(0)

            # 実際に行動したQだけを取り出す
            curr_q = torch.stack([curr_q[i].index_select(0, actions[i]) for i in range(mb_size)]).squeeze(1)

            # get next state value
            q_next = self.target_net(next_states).detach()  # (m, N_ACTIONS, N_ATOM)

            # next value mean
            q_next_mean = torch.sum(q_next * self.value_range.view(1, 1, -1), dim=2)  # (m, N_ACTIONS)
            next_actions = q_next_mean.argmax(dim=1)  # (m)
            q_next = torch.stack([q_next[i].index_select(0, next_actions[i]) for i in range(mb_size)]).squeeze(1)
            q_next = q_next.cpu().numpy()  # (m, N_ATOM)

            # categorical projection
            '''
            next_v_range : (z_j) i.e. values of possible return, shape : (m, N_ATOM)
            next_v_pos : relative position when offset of value is V_MIN, shape : (m, N_ATOM)
            '''
            # we vectorized the computation of support and position
            # rewards + γ*(1-dones)*z_j
            Tz = rewards.reshape(-1, 1) \
                  + self.gamma * (1. - dones).reshape(-1, 1) \
                  * self.value_range.cpu().numpy().reshape(1, -1)

            # clip for categorical distribution
            Tz = np.clip(Tz, self.v_min, self.v_max)

            # calc relative position of possible value: b_j
            bj = (Tz - self.v_min) / self.v_step

            # get lower/upper bound of relative position
            m_l = np.floor(bj).astype(int)   # m_l
            m_u = np.ceil(bj).astype(int)    # m_u

            # target distribution
            target_q = np.zeros((mb_size, self.n_atom))  # (m, N_ATOM)

            # we didn't vectorize the computation of target assignment.
            for i in range(mb_size):
                for j in range(self.n_atom):
                    # calc prob mass of relative position weighted with distance
                    target_q[i, m_l[i, j]] += (q_next * (m_u - bj))[i, j]
                    target_q[i, m_u[i, j]] += (q_next * (bj - m_l))[i, j]

            target_q = torch.FloatTensor(target_q).to(self.device)

            # calc huber loss, dont reduce for importance weight
            loss = - torch.sum(target_q * torch.log(curr_q + 1e-8))  # (m , N_ATOM)

            # backprop loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_target(self):
        self.pred_net.load_state_dict(self.target_net.state_dict())
