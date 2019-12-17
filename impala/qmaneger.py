import torch
import torch.multiprocessing as mp


class QManeger(object):
    """
    single-machine implementation
    """

    def __init__(self, opt, q_trace, q_batch):
        self.traces_s = []
        self.traces_a = []
        self.traces_r = []
        self.traces_d = []
        self.traces_p = []
        self.lock = mp.Lock()

        self.q_trace = q_trace
        self.q_batch = q_batch
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def listening(self):
        while True:
            traces = self.q_trace.get(block=True)
            self.traces_s.append(traces[0])
            self.traces_a.append(traces[1])
            self.traces_r.append(traces[2])
            self.traces_d.append(traces[3])
            self.traces_p.append(traces[4])

            if len(self.traces_s) > self.opt.batch_size:
                self.produce_batch()

    def produce_batch(self):
        batch_size = self.opt.batch_size
        res_s, res_a, res_r, res_d, res_p = self.traces_s[:batch_size], self.traces_a[:batch_size], \
                                            self.traces_r[:batch_size], self.traces_d[:batch_size], \
                                            self.traces_p[:batch_size]

        # delete
        del self.traces_s[:batch_size]
        del self.traces_a[:batch_size]
        del self.traces_r[:batch_size]
        del self.traces_d[:batch_size]
        del self.traces_p[:batch_size]

        res_s = torch.FloatTensor(res_s).to(self.device)
        res_a = torch.LongTensor(res_a).to(self.device)
        res_r = torch.FloatTensor(res_r).to(self.device)
        res_d = torch.FloatTensor(res_d).to(self.device)
        res_p = torch.FloatTensor(res_p).to(self.device)

        # stack batch and put
        self.q_batch.put((res_s, res_a, res_r, res_d, res_p))
