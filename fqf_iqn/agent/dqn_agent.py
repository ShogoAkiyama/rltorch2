import torch
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from model import DQN
from utils import disable_gradients, update_params, calculate_huber_loss
from .base_agent import BaseAgent


class DQNAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=5*(10**7),
                 batch_size=32, lr=5e-5, memory_size=10**6, gamma=0.99,
                 multi_step=1, update_interval=4, target_update_interval=10000,
                 start_steps=50000, epsilon_train=0.01, epsilon_eval=0.001,
                 epsilon_decay_steps=250000, double_q_learning=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=27000, grad_cliping=None, cuda=True,
                 seed=0):
        super(DQNAgent, self).__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size,
            gamma, multi_step, update_interval, target_update_interval,
            start_steps, epsilon_train, epsilon_eval, epsilon_decay_steps,
            double_q_learning, log_interval,
            eval_interval, num_eval_steps, max_episode_steps, grad_cliping,
            cuda, seed)

        # Online network.
        self.online_net = DQN(
            num_channels=env.nrow*env.ncol,
            num_actions=self.num_actions).to(self.device)
        # Target network.
        self.target_net = DQN(
            num_channels=env.nrow*env.ncol,
            num_actions=self.num_actions).to(self.device)

        # Copy parameters of the learning network to the target network.
        self.update_target()
        # Disable calculations of gradients of the target network.
        disable_gradients(self.target_net)

        self.optim = Adam(
            self.online_net.parameters(),
            lr=lr, eps=1e-2/batch_size)

    def exploit(self, state):
        # Act without randomness.
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.online_net.calculate_q(
                states=state).argmax().item()
        return action

    def learn(self):
        self.learning_steps += 1

        states, actions, rewards, next_states, dones =\
            self.memory.sample(self.batch_size)

        # Calculate features of states.
        state_embeddings = self.online_net.calculate_state_embeddings(states)

        loss = self.calculate_loss(
            state_embeddings, actions, rewards, next_states, dones)

        update_params(
            self.optim, loss,
            networks=[self.online_net],
            retain_graph=False, grad_cliping=self.grad_cliping)

        if 4*self.steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/loss', loss.detach().item(),
                4*self.steps)

            with torch.no_grad():
                q = self.online_net.calculate_q(
                    state_embeddings=state_embeddings)
            self.writer.add_scalar(
                'stats/mean_Q', q.mean().item(), 4*self.steps)

    def calculate_loss(self, state_embeddings, actions, rewards, next_states,
                       dones):

        # calculate Q(s,a)
        current_s = self.online_net.out_net(state_embeddings)
        current_sa = current_s.gather(dim=1, index=actions)

        with torch.no_grad():
            # Calculate Q values of next states.
            next_state_embeddings =\
                self.target_net.calculate_state_embeddings(next_states)
            next_q = self.target_net.calculate_q(
                state_embeddings=next_state_embeddings)

            # Calculate greedy actions.
            next_actions = torch.argmax(next_q, dim=1, keepdim=True)
            assert next_actions.shape == (self.batch_size, 1)

            # Calculate quantile values of next states and next actions.
            next_sa = next_q.gather(dim=1, index=next_actions)
            assert next_sa.shape == (self.batch_size, 1)

            # Calculate target quantile values.
            target_sa = rewards + (1.0 - dones) * self.gamma_n * next_sa
            assert target_sa.shape == (
                self.batch_size, 1)

        td_errors = (target_sa - current_sa)
        assert td_errors.shape == (self.batch_size, 1)

        loss = (td_errors**2).mean()

        return loss

    def plot(self):
        state_size = 3
        q_nrow = self.env.nrow * state_size
        q_ncol = self.env.ncol * state_size

        state = torch.FloatTensor(
                    np.eye(self.env.nrow*self.env.ncol)).to(self.device)
        with torch.no_grad():
            q_value = self.online_net.calculate_q(state).view(self.env.nrow, self.env.ncol, 4)

        value = np.zeros((q_nrow, q_ncol))

        # Left, Down, Right, Up, Center
        value[1::3, ::3] += q_value[:, :, 0].cpu().numpy()
        value[2::3, 1::3] += q_value[:, :, 1].cpu().numpy()
        value[1::3, 2::3] += q_value[:, :, 2].cpu().numpy()
        value[::3, 1::3] += q_value[:, :, 3].cpu().numpy()
        value[1::3, 1::3] += q_value.mean(dim=2).cpu().numpy()

        # ヒートマップ表示
        fig = plt.figure(figsize=(6, 12))
        ax = fig.add_subplot(1, 1, 1)
        mappable0 = plt.imshow(value, cmap=cm.jet, interpolation="bilinear",
           vmax=abs(value).max(), vmin=-abs(value).max())
        ax.set_xticks(np.arange(-0.5, q_ncol, 3))
        ax.set_yticks(np.arange(-0.5, q_nrow, 3))
        ax.set_xticklabels(range(self.env.ncol + 1))
        ax.set_yticklabels(range(self.env.nrow + 1))
        ax.grid(which="both")

        # Start: green, Goal: blue, Hole: red
        ax.plot([1], [1],  marker="o", color='g', markersize=40, alpha=0.8)
        ax.plot([1], [4],  marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
        ax.plot([1], [7],  marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
        ax.plot([1], [10], marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
        ax.plot([1], [13], marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
        ax.plot([1], [16], marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
        ax.plot([1], [19], marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
        ax.plot([1], [22], marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
        ax.plot([1], [25], marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
        ax.plot([1], [28], marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
        ax.plot([1], [31], marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
        ax.plot([1], [34], marker="o", color='r', markersize=40, alpha=0.8)
        ax.text(1, 1.3, 'START', ha='center', size=12, c='w')
        ax.text(1, 34.3, 'GOAL', ha='center', size=12, c='w')

        fig.colorbar(mappable0, ax=ax, orientation="vertical")

        plt.show()
