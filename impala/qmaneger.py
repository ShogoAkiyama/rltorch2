import torch
import torch.multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QManeger(object):
    """
    single-machine implementation
    """

    def __init__(self, opt, q_trace, q_batch):
        # traces: 0-[s, a, r, p]
        #         1-[s, a, r, p]
        # trace(s, a, rew, a_prob)
        # s[n_step+1, 3, width, height]
        # a[n_step, a_space]
        # rew[n_step]
        # a_prob[n_step]
        self.traces_s = []
        self.traces_a = []
        self.traces_r = []
        self.traces_p = []
        self.lock = mp.Lock()

        self.q_trace = q_trace
        self.q_batch = q_batch
        self.opt = opt

    def listening(self):
        while True:
            # actor put in episode
            trace = self.q_trace.get(block=True)
            # in
            self.traces_s.append(trace[0])
            self.traces_a.append(trace[1])
            self.traces_r.append(trace[2])
            self.traces_p.append(trace[3])
            # produce_batch
            if len(self.traces_s) > self.opt.batch_size:
                self.produce_batch()

    def produce_batch(self):
        batch_size = self.opt.batch_size
        # out
        res_s, res_a, res_r, res_p = self.traces_s[:batch_size], self.traces_a[:batch_size], \
                                     self.traces_r[:batch_size], self.traces_p[:batch_size]

        # delete
        del self.traces_s[:batch_size]
        del self.traces_a[:batch_size]
        del self.traces_r[:batch_size]
        del self.traces_p[:batch_size]

        # stack batch and put
        self.q_batch.put((torch.stack(res_s, dim=0).to(device), torch.stack(res_a, dim=0).to(device),
                          torch.stack(res_r, dim=0).to(device), torch.stack(res_p, dim=0).to(device)))
