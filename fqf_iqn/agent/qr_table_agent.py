import torch
import numpy as np
# import cupy as cp

from .table_base_agent import TableBaseAgent

class QRTableAgent(TableBaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=5*(10**7),
                 num_taus=32, c=0.0, lr=5e-5, gamma=0.99, multi_step=1, update_interval=4,
                 target_update_interval=10000, start_steps=50000, epsilon_train=0.01,
                 epsilon_eval=0.001, epsilon_decay_steps=250000, double_q_learning=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=27000, seed=0, cuda='cuda', sensitive=False):
        super(QRTableAgent, self).__init__(
            env, test_env, log_dir, num_steps, lr, gamma, multi_step, 
            update_interval, target_update_interval, start_steps, epsilon_train, 
            epsilon_eval, epsilon_decay_steps, double_q_learning, log_interval, 
            eval_interval, num_eval_steps, max_episode_steps, seed, cuda)

        self.num_taus = num_taus
        self.c = c
        self.sensitive = sensitive
        self.N = np.ceil(self.c*self.num_taus)

        # Online network.
        self.online_net = torch.zeros((
            self.env.nrow * self.env.ncol,
            self.num_taus, 4)).to(self.device)
        # # Target network.
        # self.target_net = torch.zeros((
        #     self.env.nrow * self.env.ncol,
        #     self.num_taus, 4)).to(self.device)

        # Copy parameters of the learning network to the target network.
        # self.update_target()

    def exploit(self, state):
        # Act without randomness.
        quantiles = self.online_net[state.argmax()]
        quantiles = quantiles.unsqueeze(0)

        probs = (quantiles[:, :, None] <= quantiles[:, None, :]
                 ).float().mean(axis=1)

        assert probs.shape == (
            1, self.num_taus, 4)

        if self.sensitive:
            q_value = (quantiles * (probs <= self.c)
                       ).sum(axis=1) / self.N
        else:
            q_value = (quantiles * (probs >= (1 - self.c))
                       ).sum(axis=1) / self.N
        action = q_value.argmax().item()
        return action

    def train_step_interval(self):
        super().train_step_interval()
        quantiles = self.online_net.clone()

        probs = (quantiles[:, :, None, ] <= quantiles[:, None, :]
                 ).float().mean(axis=1)

        if self.sensitive:
            q_value = (quantiles * (probs <= self.c)
                       ).sum(axis=1) / self.N
        else:
            q_value = (quantiles * (probs >= (1 - self.c))
                       ).sum(axis=1) / self.N

        q_value = q_value.view(self.env.nrow, self.env.ncol, 4).cpu().numpy()

        if self.steps % self.eval_interval == 0:
            self.plot(q_value)

    def learn(self, state, action, reward, next_state, done):
        self.learning_steps += 1
        N = int(self.N)
        p = torch.randint(0, self.num_taus, (N, 1)).to(self.device)
        q = torch.randint(0, self.num_taus, (N, 1)).to(self.device)
        next_action = self.online_net[state.argmax()].mean(
            axis=0).argmax().item()
        td_error = reward + (1 - done) * self.gamma \
                   * self.online_net[next_state.argmax(), :, next_action][q] \
                   - self.online_net[state.argmax(), p, action]

        self.online_net[state.argmax(), p, action] += self.lr * td_error

        # for i in range(N):
        #     td_error = reward + (1-done) * self.gamma \
        #         * self.online_net[next_state.argmax(), :, next_action][q[i]] \
        #         - self.online_net[state.argmax(), p[i], action]
        #
        #     self.online_net[state.argmax(), p[i], action] += self.lr*td_error.item()

