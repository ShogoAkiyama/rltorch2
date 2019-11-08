from common.replay_memory import EpisodeReplayMemory
from torch.utils.data.sampler import WeightedRandomSampler

import numpy as np


class R2D2ReplayMemory(EpisodeReplayMemory):

    def __init__(self, args, add_key):
        super(R2D2ReplayMemory, self).__init__(args, add_key)
        self.overlap_size = args.overlap_size
        self.seq_size = args.seq_size
        self.eta = args.eta

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
        for key in self.key_list:
            batch[key] = list()

        for key in self.key_list:
            for seq, epi in zip(seq_idx, epi_idx):
                start_v = self.overlap_size * epi
                end_v = self.overlap_size * epi + self.seq_size

                if end_v >= len(self.memory['reward'][seq]):  # out of range
                    end_v = len(self.memory['reward'][seq])
                    start_v = end_v - self.seq_size

                if key == 'state' or key == 'done':
                    end_v += self.multi_step

                if key == 'state':
                    state_list = []
                    for i in range(-3, 0):
                        ss = start_v + i
                        if ss < 0:
                            ss = 0
                        state_list.append(self.memory[key][seq][ss])
                    state_list = np.array(state_list)
                    state_list = np.concatenate((state_list, self.memory[key][seq][start_v:end_v]), axis=0)
                    batch[key].append(state_list)
                elif key == 'recc' or key == 'target_recc':
                    batch[key].append(self.memory[key][seq][start_v])
                else:
                    batch[key].append(self.memory[key][seq][start_v:end_v])
            batch[key] = np.array(batch[key])

        p = [self.memory['priority'][seq][epi] for seq, epi
             in zip(seq_idx, epi_idx)] / self.sum_p
        weights = (self.N * p) ** (-self.importance_exp)
        weights /= np.max(weights)

        return batch, seq_idx, epi_idx, weights

    def update_priority(self, td_loss, seq_idx, epi_idx):
        # td_loss = np.array(td_loss)
        priority = self.eta * np.max(td_loss, axis=1) + \
                   (1. - self.eta) * np.mean(td_loss, axis=1)
        priority = np.power(priority, self.priority_exp)

        for i, (seq, epi) in enumerate(zip(seq_idx, epi_idx)):
            self.memory['priority'][seq][epi] = priority[i]
            self.memory['sum_priority'][seq] = np.sum(self.memory['priority'][seq])
