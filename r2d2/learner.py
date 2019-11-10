import time
import numpy as np
import torch

from .model import LSTMNetwork
import torch.optim as optim
from common.utils import grad_false
from common.learner import AbstractLearner
from .replay_memory import R2D2ReplayMemory

def learner_process(args, shared_memory, shared_weights):
    learner = Learner(args, shared_memory, shared_weights)
    learner.run()

class Learner(AbstractLearner):
    def __init__(self, args, shared_memory, shared_weights):
        super(Learner, self).__init__(args, shared_weights)
        self.burn_in_size = args.burn_in_size
        self.seq_size = args.seq_size
        self.mask = np.concatenate([
            np.arange(0, self.seq_size+self.multi_step).reshape(-1, 1),
            np.arange(1, self.seq_size+self.multi_step+1).reshape(-1, 1),
            np.arange(2, self.seq_size+self.multi_step+2).reshape(-1, 1),
            np.arange(3, self.seq_size+self.multi_step+3).reshape(-1, 1)],
            axis=1)

        self.steps = 0
        self.shared_memory = shared_memory

        self.net = LSTMNetwork(self.n_state, self.n_act)
        self.target_net = LSTMNetwork(self.n_state, self.n_act)

        self.net.to(self.device)
        self.target_net.to(self.device)

        # target network
        self.target_net.eval().apply(grad_false)
        self.target_net.load_state_dict(self.net.state_dict())

        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.optim_lr)

        # memory
        add_key = ['recc', 'target_recc']
        self.memory = R2D2ReplayMemory(args, add_key)

        # model save
        self.save_model()

    def run(self):
        while len(self.memory) <= 1:#self.batch_size:
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
            seq_batch, seq_idx, epi_idx, weights = \
                self.memory.get_batch()

            # process batch
            seq_batch = self.process_batch(seq_batch)
            weights = torch.FloatTensor(weights).to(self.device).view(-1, 1)

            # burn in
            recc = seq_batch['recc']
            target_recc = seq_batch['target_recc']
            self.net.set_recc(recc[:, 0], recc[:, 1])
            self.target_net.set_recc(target_recc[:, 0], target_recc[:, 1])

            with torch.no_grad():
                _, hs, cs = \
                    self.net(seq_batch['state'][:, :self.burn_in_size+self.multi_step], return_hs_cs=True)

                _, target_hs, target_cs = \
                    self.target_net(seq_batch['state'][:, :self.burn_in_size+self.multi_step], return_hs_cs=True)

            # calculate Q and target
            batch = dict()
            for key in ['state', 'action', 'reward']:
                batch[key] = seq_batch[key][:, self.burn_in_size:self.seq_size]

            batch['done'] = \
                seq_batch['done'][:, self.burn_in_size+self.multi_step:self.seq_size+self.multi_step]
            batch['next_state'] = \
                seq_batch['state'][:, self.burn_in_size+self.multi_step:self.seq_size+self.multi_step]

            if self.burn_in_size == 0:
                batch['hs'] = recc[:, 0]
                batch['cs'] = recc[:, 1]
            else:
                batch['hs'] = hs[:, -self.multi_step-1]
                batch['cs'] = cs[:, -self.multi_step-1]
            batch['next_hs'] = hs[:, -1]
            batch['next_cs'] = cs[:, -1]
            batch['target_hs'] = target_hs[:, -1]
            batch['target_cs'] = target_cs[:, -1]

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
            self.net.reset_recc()
            self.target_net.reset_recc()
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
        batch['action'] = batch['action'].unsqueeze(2)

        self.net.set_recc(batch['hs'], batch['cs'])
        curr_q = self.net(batch['state']).gather(2, batch['action'])

        curr_q = curr_q.view(self.batch_size, -1)

        with torch.no_grad():
            self.net.set_recc(batch['next_hs'], batch['next_cs'])
            self.target_net.set_recc(batch['target_hs'], batch['target_cs'])

            next_action = torch.argmax(
                self.net(batch['next_state']), 2).unsqueeze(2)
            next_q = self.target_net(
                batch['next_state']).gather(2, next_action)

        next_q = next_q.view(self.batch_size, -1)

        # target Q
        target_q = batch['reward'] + \
                   (self.gamma ** self.multi_step) * next_q * (1.0 - batch['done'])

        return curr_q, target_q

    def process_batch(self, batch):
        batch['state'] = torch.FloatTensor(batch['state'] / 255.0).to(self.device)
        batch['state'] = batch['state'][:, self.mask].view(self.batch_size, self.seq_size+self.multi_step, self.n_state, 84, 84)
        batch['action'] = torch.LongTensor(batch['action']).to(self.device)
        batch['reward'] = torch.FloatTensor(batch['reward']).to(self.device)
        batch['done'] = torch.FloatTensor(batch['done']).to(self.device)
        batch['recc'] = torch.FloatTensor(batch['recc']).to(self.device)
        batch['target_recc'] = torch.FloatTensor(batch['target_recc']).to(self.device)
        return batch

