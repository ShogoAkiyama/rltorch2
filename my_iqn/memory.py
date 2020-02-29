import numpy as np
import random
import torch


class Memory:
    def __init__(self, args, q_trace, q_batch):
        self._storage = []

        self.q_trace = q_trace
        self.q_batch = q_batch

        self.batch_size = args.batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self._storage)

    def listening(self):
        while True:
            obs_t, action, reward, obs_tp1, done \
                = self.q_trace.get(block=True)
            data = (obs_t, action, reward, obs_tp1, done)
            self._storage.append(data)

            if len(self) > self.batch_size:
                self.produce_batch()

    def _encode_sample(self, idx):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idx:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)

        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    # def sample(self):
    #     idx = [random.randint(0, len(self._storage) - 1) for _ in range(self.batch_size)]
    #     self.q_batch.put((self._encode_sample(idx)))

    def produce_batch(self):
        idx = [random.randint(0, len(self._storage) - 1) for _ in range(self.batch_size)]
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idx:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(action)
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)

        obses_t = np.array(obses_t)
        actions = np.array(actions)
        rewards = np.array(rewards)
        obses_tp1 = np.array(obses_tp1)
        dones = np.array(dones)

        # stack batch and put
        self.q_batch.put((obses_t, actions, rewards, obses_tp1, dones))
