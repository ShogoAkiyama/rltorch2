import time
import numpy as np
import torch

from .model import QNetwork
import torch.optim as optim
from common.utils import grad_false
from common.learner import AbstractLearner

def learner_process(args, shared_memory, shared_weights):
    learner = Learner(args, shared_memory, shared_weights)
    learner.run()

class Learner(AbstractLearner):
    def __init__(self, args, shared_memory, shared_weights):
        super(Learner, self).__init__(args, shared_weights)

        self.steps = 0
        self.shared_memory = shared_memory

        self.net = QNetwork(self.n_state, self.n_act)
        self.target_net = QNetwork(self.n_state, self.n_act)

        self.net.to(self.device)
        self.target_net.to(self.device)

        # target network
        self.target_net.eval().apply(grad_false)
        self.target_net.load_state_dict(self.net.state_dict())

        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.optim_lr)

        # model save
        self.save_model()

    def run(self):
        while len(self.memory) <= self.batch_size:
            while not self.shared_memory.empty():
                batch = self.shared_memory.get()
                self.memory.load(batch)

            time.sleep(1.0)

        self.time = time.time()
        while True:
            self.epochs += 1
            self.train()
            self.interval()

    def train(self):
        self.total_loss = 0
        for epoch in range(self.update_per_epoch):
            self.steps += 1
            # sample batch
            batch, seq_idx, epi_idx, weights = \
                self.memory.get_batch()

            # process batch
            batch = self.process_batch(batch)
            weights = torch.tensor(weights, device=self.device, dtype=torch.float).view(-1, 1)

            # calculate Q and target
            curr_q, target_q = self.q_value(batch)

            # update model
            loss = torch.mean((target_q - curr_q) ** 2 * weights)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update priority
            td_loss = np.abs(target_q.cpu() - curr_q.detach().cpu()).numpy()
            self.memory.update_priority(td_loss, seq_idx, epi_idx)

            self.total_loss += loss.item()

    def evaluate(self):
        episodes = 10
        returns = np.zeros((episodes,), dtype=np.float32)
        action_bar = np.zeros(self.n_act, np.int)

        for i in range(episodes):
            state = self.env.reset()
            episode_reward = 0.
            done = False
            while not done:
                action = self.exploit(state)
                next_state, reward, done, _ = self.env.step(action)
                action_bar[action] += 1
                episode_reward += reward
                state = next_state

            returns[i] = episode_reward

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        self.writer.add_scalar(
            'reward/test', mean_return, self.steps)
        print('Learer  '
              f'Num steps: {self.steps:<5} '
              f'reward: {mean_return:<5.1f}+/- {std_return:<5.1f}')

    def exploit(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.net(state).argmax().item()
        return action

    def q_value(self, batch):
        # curr Q
        batch['action'] = batch['action'].view(-1, 1)
        curr_q = self.net(batch['state']).gather(1, batch['action'])

        with torch.no_grad():
            next_action = torch.argmax(
                self.net(batch['next_state']), 1).view(-1, 1)
            next_q = self.target_net(
                batch['next_state']).gather(1, next_action)

        # target Q
        target_q = batch['reward'].view(-1, 1) + \
                   (self.gamma ** self.multi_step) * next_q * (1.0 - batch['done'].view(-1, 1))

        return curr_q, target_q

    def process_batch(self, batch):
        batch['state'] = torch.FloatTensor(batch['state'] / 255.0).to(self.device)
        batch['action'] = torch.LongTensor(batch['action']).to(self.device)
        batch['reward'] = torch.FloatTensor(batch['reward']).to(self.device)
        batch['done'] = torch.FloatTensor(batch['done']).to(self.device)
        batch['next_state'] = torch.FloatTensor(batch['next_state'] / 255.0).to(self.device)
        return batch

