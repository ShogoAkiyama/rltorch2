import os
import time
import numpy as np
from copy import deepcopy

import torch
from torch.utils.tensorboard import SummaryWriter

from common.env import make_pytorch_env

class AbstractLearner:
    def __init__(self, args, shared_weights):

        # params
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.multi_step = args.multi_step
        self.optim_lr = args.lr
        self.update_per_epoch = args.update_per_epoch

        # gym
        self.epochs = 0

        self.model_path = os.path.join(
            './', 'logs', 'model')
        self.summary_path = os.path.join(
            './', 'logs', 'summary', 'leaner')

        # network
        self.shared_weights = shared_weights
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.env = make_pytorch_env(args.env_id)
        self.n_state = self.env.observation_space.shape[0]
        self.n_act = self.env.action_space.n

        # interval process
        self.eval_interval = args.learner_eval
        self.load_memory_interval = args.learner_load_memory
        self.save_model_interval = args.learner_save_model
        self.target_update_interval = args.learner_target_update
        self.save_log_interval = args.learner_save_log

        # summary
        self.writer = SummaryWriter(log_dir=self.summary_path)

    def save_model(self):
        self.shared_weights['net_state'] = deepcopy(self.net).cpu().state_dict()
        self.shared_weights['target_net_state'] = deepcopy(self.target_net).cpu().state_dict()

    def interval(self):
        if self.epochs % self.eval_interval == 0:
            self.evaluate()

        if self.epochs % self.load_memory_interval == 0:
            while not self.shared_memory.empty():
                batch = self.shared_memory.get()
                self.memory.load(batch)
            # self.memory.load()

        if self.epochs % self.save_model_interval == 0:
            self.save_model()

        if self.epochs % self.save_model_interval*100 == 0:
            self.net.save(self.model_path)
            self.target_net.save(self.model_path, target=True)

        if self.epochs % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.net.state_dict())
            self.target_net.eval()

        self.log()

    def log(self):
        now = time.time()
        if self.epochs % self.save_log_interval == 0:
            self.writer.add_scalar(
                "loss/learner",
                self.total_loss,
                self.epochs)
        print(
            f"Learer: loss: {self.total_loss:< 8.3f} "
            f"memory: {len(self.memory):<5} \t"
            f"time: {now - self.time:.2f}s")
        self.time = now

    def __del__(self):
        print(
            "----------------------------------------\n"
            "!! Learner: PROCESS HAS BEEN TERMINATED.\n"
            "----------------------------------------")
        self.writer.close()
