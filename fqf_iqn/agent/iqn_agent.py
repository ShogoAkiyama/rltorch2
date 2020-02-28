import torch
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from model import IQN
from utils import disable_gradients, update_params,\
    calculate_quantile_huber_loss, evaluate_quantile_at_action
from .base_agent import BaseAgent


class IQNAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=5*(10**7),
                 batch_size=32, N=64, N_dash=64, K=32, num_cosines=64,
                 kappa=1.0, lr=5e-5, memory_size=10**6, gamma=0.99,
                 multi_step=1, update_interval=4, target_update_interval=10000,
                 start_steps=50000, epsilon_train=0.01, epsilon_eval=0.001,
                 epsilon_decay_steps=250000, double_q_learning=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=27000, grad_cliping=None, cuda=True,
                 seed=0):
        super(IQNAgent, self).__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size,
            gamma, multi_step, update_interval, target_update_interval,
            start_steps, epsilon_train, epsilon_eval, epsilon_decay_steps,
            double_q_learning, log_interval,
            eval_interval, num_eval_steps, max_episode_steps, grad_cliping,
            cuda, seed)

        # Online network.
        self.online_net = IQN(
            num_channels=env.observation_space.n,
            num_actions=self.num_actions, K=K, num_cosines=num_cosines).to(self.device)
        # Target network.
        self.target_net = IQN(
            num_channels=env.observation_space.n,
            num_actions=self.num_actions, K=K, num_cosines=num_cosines).to(self.device)

        # Copy parameters of the learning network to the target network.
        self.update_target()
        # Disable calculations of gradients of the target network.
        disable_gradients(self.target_net)

        self.optim = Adam(
            self.online_net.parameters(),
            lr=lr, eps=1e-2/batch_size)

        self.N = N
        self.N_dash = N_dash
        self.K = K
        self.num_cosines = num_cosines
        self.kappa = kappa

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

        quantile_loss = self.calculate_loss(
            state_embeddings, actions, rewards, next_states, dones)

        update_params(
            self.optim, quantile_loss,
            networks=[self.online_net],
            retain_graph=False, grad_cliping=self.grad_cliping)

        if 4*self.steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/quantile_loss', quantile_loss.detach().item(),
                4*self.steps)

            with torch.no_grad():
                q = self.online_net.calculate_q(
                    state_embeddings=state_embeddings)
            self.writer.add_scalar(
                'stats/mean_Q', q.mean().item(), 4*self.steps)

    def calculate_loss(self, state_embeddings, actions, rewards, next_states,
                       dones):

        # Sample fractions.
        taus = torch.rand(
            self.batch_size, self.N, dtype=state_embeddings.dtype,
            device=state_embeddings.device)

        # Calculate quantile values of current states and all actions.
        current_s_quantiles = self.online_net.calculate_quantiles(
            taus, state_embeddings=state_embeddings)
        assert current_s_quantiles.shape == (
            self.batch_size, self.N, self.num_actions)

        # Calculate quantile values of current states and current actions.
        current_sa_quantiles = evaluate_quantile_at_action(
            current_s_quantiles, actions)
        assert current_sa_quantiles.shape == (self.batch_size, self.N, 1)

        with torch.no_grad():
            # Calculate Q values of next states.
            if self.double_q_learning:
                next_q = self.online_net.calculate_q(states=next_states)
            else:
                next_state_embeddings =\
                    self.target_net.calculate_state_embeddings(next_states)
                next_q = self.target_net.calculate_q(
                    state_embeddings=next_state_embeddings)

            # Calculate greedy actions.
            next_actions = torch.argmax(next_q, dim=1, keepdim=True)
            assert next_actions.shape == (self.batch_size, 1)

            # Calculate features of next states.
            if self.double_q_learning:
                next_state_embeddings =\
                    self.target_net.calculate_state_embeddings(next_states)

            # Sample next fractions.
            tau_dashes = torch.rand(
                self.batch_size, self.N_dash, dtype=state_embeddings.dtype,
                device=state_embeddings.device)

            # Calculate quantile values of next states and all actions.
            next_s_quantiles = self.target_net.calculate_quantiles(
                tau_dashes, state_embeddings=next_state_embeddings)

            # Calculate quantile values of next states and next actions.
            next_sa_quantiles = evaluate_quantile_at_action(
                next_s_quantiles, next_actions).transpose(1, 2)
            assert next_sa_quantiles.shape == (self.batch_size, 1, self.N_dash)

            # Calculate target quantile values.
            target_sa_quantiles = rewards[..., None] + (
                1.0 - dones[..., None]) * self.gamma_n * next_sa_quantiles
            assert target_sa_quantiles.shape == (
                self.batch_size, 1, self.N_dash)

        td_errors = target_sa_quantiles - current_sa_quantiles
        assert td_errors.shape == (self.batch_size, self.N, self.N_dash)

        quantile_huber_loss = calculate_quantile_huber_loss(
            td_errors, taus, self.kappa)

        return quantile_huber_loss

    def plot(self):
        state = torch.FloatTensor(
                    np.eye(self.env.observation_space.n)).to(self.device)
        with torch.no_grad():
            q_value = self.online_net.calculate_q(state).view(4, 4, 4)

        value = np.zeros((12, 12))

        # Left, Down, Right, Up, Center
        value[1::3, ::3] += q_value[:, :, 0].cpu().numpy()
        value[2::3, 1::3] += q_value[:, :, 1].cpu().numpy()
        value[1::3, 2::3] += q_value[:, :, 2].cpu().numpy()
        value[::3, 1::3] += q_value[:, :, 3].cpu().numpy()
        value[1::3, 1::3] += q_value.mean(dim=2).cpu().numpy()

        # ヒートマップ表示
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(1, 1, 1)
        mappable0 = plt.imshow(value, cmap=cm.jet, interpolation="bilinear",
           vmax=abs(value).max(), vmin=-abs(value).max())
        ax.set_xticks(np.arange(-0.5, 12, 3))
        ax.set_yticks(np.arange(-0.5, 12, 3))
        ax.set_xticklabels(range(4 + 1))
        ax.set_yticklabels(range(4 + 1))
        ax.grid(which="both")

        # Start: green, Goal: blue, Hole: red
        ax.plot([1], [1], marker="o", color='g', markersize=40, alpha=0.8)
        ax.plot([1], [10], marker="o", color='r', markersize=40, alpha=0.8)
        ax.plot([1], [4], marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
        ax.plot([1], [7], marker="x", color='b', markersize=30, markeredgewidth=10, alpha=0.8)
        ax.text(1, 1.3, 'START', ha='center', size=12, c='w')
        ax.text(1, 10.3, 'GOAL', ha='center', size=12, c='w')

        fig.colorbar(mappable0, ax=ax, orientation="vertical")

        plt.show()
