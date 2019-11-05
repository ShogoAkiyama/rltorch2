import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler
import os
import gc
import fasteners
import pickle

gc.enable()

class EpisodeReplayMemory:
    def __init__(self, args):
        # args
        self.n_actors = args.n_actors
        self.priority_exp = args.priority_exp
        self.importance_exp = args.importance_exp

        self.batch_size = args.batch_size
        self.multi_step = args.multi_step

        # path
        self.memory_path = os.path.join(
            './', 'logs', 'memory')
        self.memory_size = args.memory_size

        self.key_list = ['state', 'action', 'reward', 'done']

        self.memory = {}
        for key in self.key_list + ['priority', 'sum_priority']:
            self.memory[key] = list()

        # priority
        self.N = 0
        self.sum_p = 0

    def __len__(self):
        return len(self.memory["sum_priority"])

    def reset(self):
        self.memory = {}
        for key in self.key_list + ['priority', 'sum_priority']:
            self.memory[key] = list()

        self.N = 0
        self.sum_p = 0

    def get(self):
        return self.memory

    def load(self, memory):
        for key in self.key_list + ['priority', 'sum_priority']:
            self.memory[key].extend(memory[key])

        self.N += sum(len(v) for v in memory['priority'])
        self.sum_p += sum([pri for pri in memory['sum_priority']])

        if len(self.memory['sum_priority']) > self.memory_size:
            # calc priority
            memory = dict()
            memory['priority'] = \
                self.memory['priority'][:len(self.memory['priority']) - int(self.memory_size)]
            memory['sum_priority'] = \
                self.memory['sum_priority'][:len(self.memory['sum_priority']) - int(self.memory_size)]

            self.N -= sum(len(v) for v in memory['priority'])
            self.sum_p -= sum([pri for pri in memory['sum_priority']])

            # update memory
            for key in self.key_list + ['priority', 'sum_priority']:
                self.memory[key] = self.memory[key][int(-self.memory_size):]

    # def save(self, actor_id):
    #     memory_path = os.path.join(self.memory_path, f'memory{actor_id}.pt')
    #     for key in self.key_list + ['priority', 'sum_priority']:
    #         self.memory[key] = list(self.memory[key])
    #
    #     lock = fasteners.InterProcessLock(memory_path)
    #     gotten = lock.acquire(blocking=False)
    #     try:
    #         if gotten:
    #             if os.path.isfile(memory_path) and os.path.getsize(memory_path) > 0:
    #                 # pass
    #                 memory = torch.load(
    #                     memory_path, map_location=lambda storage, loc: storage)
    #                 for key in self.key_list + ['priority', 'sum_priority']:
    #                     memory[key].extend(self.memory[key])
    #                 torch.save(memory, memory_path)
    #             else:
    #                 memory = dict()
    #                 for key in self.key_list + ['priority', 'sum_priority']:
    #                     memory[key] = self.memory[key]
    #                 torch.save(memory, memory_path)
    #     finally:
    #         if gotten:
    #             lock.release()
    #
    #     self.memory = {}
    #     for key in self.key_list + ['priority', 'sum_priority']:
    #         self.memory[key] = list()

    # def load(self):
    #     for actor_id in range(self.n_actors):
    #         memory_path = os.path.join(self.memory_path, f'memory{actor_id}.pt')
    #         is_memory = os.path.isfile(memory_path)
    #         if is_memory:
    #             lock = fasteners.InterProcessLock(memory_path)
    #             gotten = lock.acquire(blocking=False)
    #             try:
    #                 if gotten and os.path.isfile(memory_path) and os.path.getsize(memory_path) > 200:
    #                     memory = torch.load(
    #                         memory_path, map_location=lambda storage, loc: storage)
    #                     for key in self.key_list + ['priority', 'sum_priority']:
    #                         self.memory[key].extend(memory[key])
    #                     N = sum(len(v) for v in memory['priority'])
    #                     sum_p = sum([pri for pri in memory['sum_priority']])
    #                     self.N += N
    #                     self.sum_p += sum_p
    #
    #                     if len(self.memory['sum_priority']) > self.memory_size:
    #                         # calc priority
    #                         memory = dict()
    #                         memory['priority'] = \
    #                             self.memory['priority'][:len(self.memory['priority']) - int(self.memory_size)]
    #                         memory['sum_priority'] = \
    #                             self.memory['sum_priority'][:len(self.memory['sum_priority']) - int(self.memory_size)]
    #
    #                         self.N -= sum(len(v) for v in memory['priority'])
    #                         self.sum_p -= sum([pri for pri in memory['sum_priority']])
    #
    #                         # update memory
    #                         for key in self.key_list + ['priority', 'sum_priority']:
    #                             self.memory[key] = self.memory[key][int(-self.memory_size):]
    #
    #                         gc.collect()
    #
    #             except pickle.UnpicklingError:
    #                 print("Pickle data was truncated.")
    #                 os.remove(memory_path)
    #             finally:
    #                 if gotten:
    #                     if os.path.isfile(memory_path):
    #                         os.remove(memory_path)
    #                     lock.release()

    def add(self, episode_buff):
        for key in self.key_list + ['priority']:
            self.memory[key].append(episode_buff[key])
        self.memory['sum_priority'].append(np.sum(episode_buff['priority']))

    def get_batch(self):
        # get sequence idx
        sampler = WeightedRandomSampler(
            self.memory['sum_priority'], self.batch_size)
        seq_idx = list(sampler)

        # get episode idx
        epi_idx = []
        for seq in seq_idx:
            sampler = WeightedRandomSampler(self.memory['priority'][seq], 1)
            epi_idx.append(list(sampler)[0])

        # get batch
        batch = {}
        for key in self.key_list + ['next_state']:
            batch[key] = list()

        for seq, epi in zip(seq_idx, epi_idx):
            next_idx = epi + self.multi_step + 1

            for key in ['state']:
                state_list = []
                for i in range(-3, 1):   # -3 ~ 0
                    ss = epi + i
                    if ss < 0:
                        ss = 0
                    state_list.append(self.memory[key][seq][ss])
                state_list = np.concatenate(state_list, axis=0)
                batch[key].append(state_list)

            batch['action'].append(self.memory['action'][seq][epi])
            batch['reward'].append(self.memory['reward'][seq][epi])

            batch['next_state'].append(
                np.concatenate(self.memory['state'][seq][next_idx-4:next_idx], axis=0))
            batch['done'].append(self.memory['done'][seq][next_idx-1])

        for key in self.key_list + ['next_state']:
            batch[key] = np.array(batch[key])

        p = [self.memory['priority'][seq][epi] for seq, epi
             in zip(seq_idx, epi_idx)] / self.sum_p
        weights = (self.N * p) ** (-self.importance_exp)
        weights /= np.max(weights)

        return batch, seq_idx, epi_idx, weights

    def update_priority(self, td_loss, seq_idx, epi_idx):
        priority = td_loss
        priority = np.power(priority, self.priority_exp)

        for i, (seq, epi) in enumerate(zip(seq_idx, epi_idx)):
            self.memory['priority'][seq][epi] = priority[i][0]
            self.memory['sum_priority'][seq] = np.sum(self.memory['priority'][seq])
