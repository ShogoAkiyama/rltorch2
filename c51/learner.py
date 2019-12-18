import numpy as np
import torch
from model import ConvNet
from wrappers import make_pytorch_env

class Learner:
    def __init__(self, args):
        self.learn_step_counter = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.env = make_pytorch_env(args.env)
        self.n_act = self.env.action_space.n
        self.n_state = self.env.observation_space.shape[0]
        self.n_atom = args.atom

        self.pred_net = ConvNet(self.n_state, self.n_act, self.n_atom).to(self.device)
        self.target_net = ConvNet(self.n_state, self.n_act, self.n_atom).to(self.device)


    def learn(self):
        self.learn_step_counter += 1
        # target parameter update
        if self.learn_step_counter % 1 == 0:
            self.update_target()

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        b_w, b_idxes = np.ones_like(rewards), None

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        # action value distribution prediction
        q_eval = self.pred_net(states)  # (m, N_ACTIONS, N_ATOM)
        mb_size = q_eval.size(0)
        q_eval = torch.stack([q_eval[i].index_select(0, actions[i]) for i in range(mb_size)]).squeeze(1)
        # (m, N_ATOM)

        # target distribution
        q_target = np.zeros((mb_size, self.n_atom))  # (m, N_ATOM)

        # get next state value
        q_next = self.target_net(next_states).detach()  # (m, N_ACTIONS, N_ATOM)
        # next value mean
        q_next_mean = torch.sum(q_next * self.value_range.view(1, 1, -1), dim=2)  # (m, N_ACTIONS)
        best_actions = q_next_mean.argmax(dim=1)  # (m)
        q_next = torch.stack([q_next[i].index_select(0, best_actions[i]) for i in range(mb_size)]).squeeze(1)
        q_next = q_next.data.cpu().numpy()  # (m, N_ATOM)

        # categorical projection
        '''
        next_v_range : (z_j) i.e. values of possible return, shape : (m, N_ATOM)
        next_v_pos : relative position when offset of value is V_MIN, shape : (m, N_ATOM)
        '''
        # we vectorized the computation of support and position
        next_v_range = np.expand_dims(rewards, 1) + self.gamma * np.expand_dims((1. - dones), 1) \
                       * np.expand_dims(self.value_range.data.cpu().numpy(), 0)
        next_v_pos = np.zeros_like(next_v_range)
        # clip for categorical distribution
        next_v_range = np.clip(next_v_range, self.v_min, self.v_max)
        # calc relative position of possible value
        next_v_pos = (next_v_range - self.v_min) / self.v_step
        # get lower/upper bound of relative position
        lb = np.floor(next_v_pos).astype(int)
        ub = np.ceil(next_v_pos).astype(int)
        # we didn't vectorize the computation of target assignment.
        for i in range(mb_size):
            for j in range(self.n_atom):
                # calc prob mass of relative position weighted with distance
                q_target[i, lb[i, j]] += (q_next * (ub - next_v_pos))[i, j]
                q_target[i, ub[i, j]] += (q_next * (next_v_pos - lb))[i, j]

        q_target = torch.FloatTensor(q_target).to(self.device)

        # calc huber loss, dont reduce for importance weight
        loss = q_target * (- torch.log(q_eval + 1e-8))  # (m , N_ATOM)
        loss = torch.mean(loss)

        # calc importance weighted loss
        b_w = torch.Tensor(b_w).to(self.device)

        loss = torch.mean(b_w * loss)

        # backprop loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.pred_net.load_state_dict(self.target_net.state_dict())

