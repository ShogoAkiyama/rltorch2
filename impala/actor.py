import torch
from torch.optim import Adam
from model import ActorCritic
import retro
from torchvision import transforms
import numpy as np

from common.env import make_pytorch_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(48),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=(0.5, 0.5, 0.5),
                                    std=(0.5, 0.5, 0.5))])

class Actor(object):
    def __init__(self, opt, q_trace, learner):
        self.opt = opt
        self.q_trace = q_trace
        self.learner = learner

        # 游戏
        self.env = None
        # s_channel = self.env.observation_space.shape[0]
        # a_space = self.env.action_space

        # 网络
        self.behaviour = ActorCritic(opt).to(device)

    def performing(self, rank):
        torch.manual_seed(self.opt.seed)
        # 每个线程初始化环境
        self.env = retro.make(game=self.opt.env)
        self.env.seed(self.opt.seed + rank)

        s = self.env.reset()
        s = transform(s).unsqueeze(dim=0).to(device)
        episode_length = 0
        r_sum = 0.
        done = True
        while True:
            # apply
            # print(type(self.learner.network.state_dict()))
            self.behaviour.load_state_dict(self.learner.network.state_dict())
            # LSTM
            if done:
                cx = torch.zeros(1, 256).to(device)
                hx = torch.zeros(1, 256).to(device)
            else:
                cx = cx.detach()
                hx = hx.detach()

            trace_s, trace_a, trace_rew, trace_aprob = [], [], [], []
            # collect n-step
            for n in range(self.opt.n_step):
                episode_length += 1
                #  add to trace - 0
                trace_s.append(s)
                value, logit, (hx, cx) = self.behaviour((s, (hx, cx)))
                logit = logit.detach()
                action = torch.bernoulli(logit)

                s, rew, done, info = self.env.step(action.squeeze().to("cpu").numpy().astype(np.int8))
                r_sum += rew
                s = transform(s).unsqueeze(dim=0).to(device)
                rew = torch.Tensor([rew]).to(device)
                done = done or episode_length >= self.opt.max_episode_length

                #  add to trace - 1
                trace_a.append(action)
                trace_rew.append(rew)
                trace_aprob.append(logit)
                if done:
                    print("over, reward {}".format(r_sum))
                    r_sum = 0
                    episode_length = 0
                    # game over punishment
                    trace_rew[-1] = torch.Tensor([-200.]).to(device)
                    break
            # add to trace - 2
            trace_s.append(s)
            # stack n-step
            # s[n_step+1, 3, width, height]
            # a[n_step, a_space]
            # rew[n_step]
            # a_prob[n_step]
            trace_s = torch.cat(tuple(trace_s), dim=0)
            zeros = torch.zeros((self.opt.n_step + 1,) + trace_s.size()[1:]).to(device)  # expand
            zeros[:trace_s.size(0)] += trace_s
            trace_s = zeros

            trace_a = torch.cat(tuple(trace_a), dim=0)
            zeros = torch.zeros((self.opt.n_step,) + trace_a.size()[1:]).to(device)  # expand
            zeros[:trace_a.size(0)] += trace_a
            trace_a = zeros

            trace_rew = torch.cat(tuple(trace_rew), dim=0)
            zeros = torch.zeros(self.opt.n_step).to(device)  # expand
            zeros[:trace_rew.size(0)] += trace_rew
            trace_rew = zeros

            trace_aprob = torch.cat(tuple(trace_aprob), dim=0)
            zeros = torch.zeros((self.opt.n_step,) + trace_aprob.size()[1:]).to(device)  # expand
            zeros[:trace_aprob.size(0)] += trace_aprob
            trace_aprob = zeros

            # submit trace to queue
            self.q_trace.put((trace_s.to("cpu"), trace_a.to("cpu"), trace_rew.to("cpu"), trace_aprob.to("cpu")), block=True)

            if done:
                s = self.env.reset()
                s = transform(s).unsqueeze(dim=0).to(device)