import torch
from torch.optim import Adam
from model import ActorCritic
import retro
from torchvision import transforms
import numpy as np

from env import make_pytorch_env

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(48),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=(0.5, 0.5, 0.5),
                                    std=(0.5, 0.5, 0.5))])

class Actor(object):
    def __init__(self, opt, q_trace, shared_weights):
        self.opt = opt
        self.q_trace = q_trace
        self.shared_weights = shared_weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = None
        self.n_state = None
        self.n_act = None

        self.episodes = 0

        # モデル
        self.net = ActorCritic(opt).to(self.device)

    def performing(self, rank):
        torch.manual_seed(self.opt.seed)

        self.env = make_pytorch_env(self.opt.env)
        self.env.seed(self.opt.seed + rank)
        self.n_state = self.env.observation_space.shape
        self.n_act = self.env.action_space.n

        while True:
            self.load_model()
            self.train_episode()

    def train_episode(self):
        self.episodes += 1
        episode_steps = 0
        episode_reward = 0
        done = False
        state = self.env.reset()
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        self.net.reset_recc()

        while not done:
            trace_s = torch.zeros((self.opt.n_step + 1,) + self.n_state).to(self.device)
            trace_a = torch.zeros(self.opt.n_step).to(self.device)
            trace_r = torch.zeros(self.opt.n_step).to(self.device)
            trace_aprob = torch.zeros((self.opt.n_step, self.n_act)).to(self.device)

            # collect n-step
            for i in range(self.opt.n_step):
                episode_steps += 1
                #  add to trace - 0
                trace_s[i] = state.detach()
                value, logit = self.net(state)
                logit = logit.detach()
                action = torch.exp(logit).argmax(dim=-1)

                state, reward, done, info = self.env.step(action.squeeze().to("cpu").numpy().astype(np.int8))
                episode_reward += reward
                state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                # state = transform(state).unsqueeze(dim=0).to(self.device)
                reward = torch.Tensor([reward]).to(self.device)
                done = done or episode_steps >= self.opt.max_episode_length

                # add to trace - 1
                trace_a[i] = action
                trace_r[i] = reward
                trace_aprob[i] = logit
                if done:
                    # game over punishment
                    # trace_r[-1] = torch.Tensor([-200.]).to(self.device)
                    break

            # add to trace - 2
            trace_s[i] = state

            # submit trace to queue
            # [state, action, reward. action prob]
            self.q_trace.put((trace_s.to("cpu"), trace_a.to("cpu"), trace_r.to("cpu"), trace_aprob.to("cpu")), block=True)

        print(f'episode: {self.episodes:<4}  '
              f'episode steps: {episode_steps:<4}  '
              f'reward: {episode_reward:<5.1f}')

    def load_model(self):
        try:
            self.net.load_state_dict(self.shared_weights['net_state'])
            # self.target_net.load_state_dict(self.shared_weights['target_net_state'])
        except:
            print('load error')
