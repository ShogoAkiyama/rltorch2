import torch
import torch.multiprocessing as mp
import random

class QManeger(object):

    def __init__(self, opt, q_trace, q_batch):
        self.traces_s = []
        self.traces_a = []
        self.traces_r = []
        self.lock = mp.Lock()

        self.q_trace = q_trace
        self.q_batch = q_batch
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _push_one(self, state, action, reward):
        self.traces_s.append(state)
        self.traces_a.append(action)
        self.traces_r.append(reward)

    def listening(self):
        while True:
            traces = self.q_trace.get(block=True)
            for s, a, r in zip(traces[0], traces[1], traces[2]):
                self._push_one(s, a, r)

            if len(self.traces_s) > self.opt.batch_size:
                self.produce_batch()

    def produce_batch(self):
        batch_size = self.opt.batch_size

        res_s, res_a, res_r = self.traces_s[:batch_size], self.traces_a[:batch_size], \
                              self.traces_r[:batch_size]

        # delete
        del self.traces_s[:batch_size]
        del self.traces_a[:batch_size]
        del self.traces_r[:batch_size]

        res_s = torch.FloatTensor(res_s).to(self.device)
        res_a = torch.LongTensor(res_a).to(self.device)
        res_r = torch.FloatTensor(res_r).to(self.device).view(-1, 1)

        # stack batch and put
        self.q_batch.put((res_s, res_a, res_r))
