import numpy as np
import random
import torch


class QManeger:
    def __init__(self, args, q_trace, q_batch):
        self._storage = []

        self.q_trace = q_trace
        self.q_batch = q_batch

        self.batch_size = args.batch_size
        self.device = args.device

    def __len__(self):
        return len(self._storage)

    def listening(self):
        while True:
            state, action, reward, next_state, done \
                    = self.q_trace.get(block=True)
            data = (state, action, reward, next_state, done)
            self._storage.append(data)

            if len(self) > self.batch_size:
                self.produce_batch()

    def produce_batch(self):
        idx = [random.randint(0, len(self._storage) - 1) for _ in range(self.batch_size)]
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in idx:
            data = self._storage[i]
            state, action, reward, next_state, done = data
            states.append(np.array(state, copy=False))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)

        # stack batch and put
        self.q_batch.put((states, actions, rewards, next_states, dones))
