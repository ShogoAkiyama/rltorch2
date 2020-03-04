import torch

from .table_base_agent import TableBaseAgent
from fqf_iqn.memory import LazyMultiStepMemory


class QRDQNAgent(TableBaseAgent):
    def __init__(self, env, test_env, log_dir, num_steps=5*(10**7),
                 batch_size=32, num_taus=32, num_cosines=64,
                 kappa=1.0, quantile_lr=5e-5,
                 memory_size=10**6, gamma=0.99, multi_step=1,
                 update_interval=4, target_update_interval=10000,
                 start_steps=50000, epsilon_train=0.01, epsilon_eval=0.001,
                 epsilon_decay_steps=250000, double_q_learning=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=27000, cuda=True,
                 seed=0):
        super(QRDQNAgent, self).__init__(
            env, test_env, log_dir, num_steps, quantile_lr, gamma, multi_step,
            update_interval, target_update_interval, start_steps, epsilon_train,
            epsilon_eval, epsilon_decay_steps, double_q_learning, log_interval,
            eval_interval, num_eval_steps, max_episode_steps, seed, cuda)

        self.batch_size = batch_size
        self.num_taus = num_taus
        self.kappa = kappa
        self.num_actions = 4

        # Replay memory which is memory-efficient to store stacked frames.
        self.memory = LazyMultiStepMemory(
            memory_size, 1,
            self.device, gamma, multi_step)

        # Online network.
        self.online_net = torch.zeros((
            self.env.nrow * self.env.ncol, self.num_taus, self.num_actions),
            device=self.device, requires_grad=True)

        # Online network.
        self.target_net = torch.zeros((
            self.env.nrow * self.env.ncol, self.num_taus, self.num_actions),
            device=self.device)

        self.update_target()

        self.tau_hats = self.calculate_tau_hats()

    def calculate_tau_hats(self):
        # taus value: [0, 1/N, ..., N/N], shape: [batch_size, num_taus+1]
        taus = torch.arange(
            0, self.num_taus+1
            ).to(self.device).repeat(self.batch_size, 1).float() / self.num_taus

        assert taus.shape == (self.batch_size, self.num_taus+1)

        tau_hats = (taus[:, :-1] + taus[:, 1:]).detach() / 2.

        return tau_hats

    def train_step_interval(self):
        super().train_step_interval()

        if self.steps % self.target_update_interval == 0:
            self.update_target()

        with torch.no_grad():
            quantiles = self.online_net.clone()
            q_value = quantiles.mean(axis=1)
            q_value = q_value.view(self.env.nrow, self.env.ncol, self.num_actions).cpu().numpy()

        if self.is_update():
            self.learn()

        if self.steps % self.eval_interval == 0:
            print("plot")
            self.plot(q_value)

    def update_target(self):
        # print(self.online_net[0])
        self.target_net = self.online_net.clone().detach()
        # print(self.target_net[0])

    def exploit(self, state):
        state = torch.LongTensor([state]).to(self.device)

        with torch.no_grad():
            quantiles = self.calculate_q(state)
            assert quantiles.shape == (
                1, self.num_actions)
            action = quantiles.argmax().item()

        return action

    def calculate_q(self, states=None):
        assert states is not None

        batch_size = states.shape[0]

        # Calculate quantiles.
        quantile = self.online_net[states]
        assert quantile.shape == (
            batch_size, self.num_taus, self.num_actions)

        # Calculate expectations of value distribution.
        q = quantile.sum(dim=1) / self.num_taus

        assert q.shape == (batch_size, self.num_actions)

        return q

    def calculate_target_q(self, states=None):
        assert states is not None

        batch_size = states.shape[0]

        # Calculate quantiles.
        quantile = self.target_net[states]
        # print(states.shape)
        assert quantile.shape == (
            batch_size, self.num_taus, self.num_actions)

        # Calculate expectations of value distribution.
        q = quantile.sum(dim=1) / self.num_taus

        assert q.shape == (batch_size, self.num_actions)

        return q

    def train_episode(self):
        self.episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = False
        state = self.env.reset()

        while (not done) and episode_steps <= self.max_episode_steps:

            if self.is_greedy(eval=False):
                action = self.explore()
            else:
                action = self.exploit(state)

            next_state, reward, done, _ = self.env.step(action)

            self.memory.append(
                state, action, reward, next_state, done)

            self.steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state

            self.train_step_interval()

        # We log running mean of stats.
        self.train_return.append(episode_return)

        # We log evaluation results along with training frames = 4 * steps.
        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'return/train', self.train_return.get(), 4 * self.steps)

        if self.episodes % 1000 == 0:
            print(f'Episode: {self.episodes:<4}  '
                  f'episode steps: {episode_steps:<4}  '
                  f'return: {episode_return:<5.1f}')

    def learn(self):
        self.learning_steps += 1

        states, actions, rewards, next_states, dones =\
            self.memory.sample(self.batch_size)

        # # Calculate embeddings of current states.
        # print(states.shape, ' ', actions.shape)
        current_sa_quantile_hats = evaluate_quantile_at_action(
            self.online_net[states], actions)
        assert current_sa_quantile_hats.shape == (
                self.batch_size, self.num_taus, 1)

        quantile_loss = self.calculate_quantile_loss(
            current_sa_quantile_hats, rewards, next_states, dones)

        quantile_loss.backward()
        with torch.no_grad():
            self.online_net -= self.lr * self.online_net.grad
            self.online_net.grad.zero_()

        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/quantile_loss', quantile_loss.detach().item(),
                4*self.steps)

            with torch.no_grad():
                mean_q = self.calculate_q(states=states).mean()

            self.writer.add_scalar(
                'stats/mean_Q', mean_q, 4*self.steps)

    def calculate_quantile_loss(self, current_sa_quantile_hats, 
                                rewards, next_states, dones):

        with torch.no_grad():
            # Calculate Q values of next states.
            if self.double_q_learning:
                next_q = self.calculate_q(states=next_states)
            else:
                next_q = self.calculate_target_q(states=next_states)

            # Calculate greedy actions.
            next_actions = torch.argmax(next_q, dim=1, keepdim=True)
            assert next_actions.shape == (self.batch_size, 1)

            # Calculate quantile values of next states and actions at tau_hats.
            next_sa_quantile_hats = evaluate_quantile_at_action(
                self.target_net[next_states], next_actions).transpose(1, 2)

            assert next_sa_quantile_hats.shape == (
                self.batch_size, 1, self.num_taus)

            # Calculate target quantile values.
            target_sa_quantile_hats = rewards[..., None] + (
                1.0 - dones[..., None]) * self.gamma * next_sa_quantile_hats
            assert target_sa_quantile_hats.shape == (
                self.batch_size, 1, self.num_taus)

        td_errors = target_sa_quantile_hats - current_sa_quantile_hats
        assert td_errors.shape == (
            self.batch_size, self.num_taus, self.num_taus)

        quantile_huber_loss = calculate_quantile_huber_loss(
            td_errors, self.tau_hats, self.kappa)

        return quantile_huber_loss


def evaluate_quantile_at_action(s_quantiles, actions):
    assert s_quantiles.shape[0] == actions.shape[0]

    batch_size = s_quantiles.shape[0]
    num_taus = s_quantiles.shape[1]

    # Expand actions into (batch_size, num_taus, 1).
    action_index = actions[..., None].expand(batch_size, num_taus, 1)

    # Calculate quantile values at specified actions.
    sa_quantiles = s_quantiles.gather(dim=2, index=action_index)

    return sa_quantiles


def calculate_huber_loss(td_errors, kappa=1.0):
    return torch.where(
        td_errors.abs() <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (td_errors.abs() - 0.5 * kappa))


def calculate_quantile_huber_loss(td_errors, taus, kappa=1.0):
    assert not taus.requires_grad
    batch_size, num_taus, num_target_taus = td_errors.shape

    # Calculate huber loss element-wisely.
    element_wise_huber_loss = calculate_huber_loss(td_errors, kappa)
    assert element_wise_huber_loss.shape == (
        batch_size, num_taus, num_target_taus)

    # Calculate quantile huber loss element-wisely.
    element_wise_quantile_huber_loss = torch.abs(
        taus[..., None] - (td_errors.detach() < 0).float()
        ) * element_wise_huber_loss / kappa
    assert element_wise_quantile_huber_loss.shape == (
        batch_size, num_taus, num_target_taus)

    return element_wise_quantile_huber_loss.sum(dim=1).mean()
