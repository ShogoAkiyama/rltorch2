from model import ConvNet
from wrappers import make_pytorch_env
import torch
import numpy as np
import time
from collections import deque


class Actor:
    def __init__(self, args, actor_id, q_trace, learner):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_trace = q_trace
        self.learner = learner

        self.env = make_pytorch_env(args.env)
        self.n_act = self.env.action_space.n
        self.n_state = self.env.observation_space.shape[0]
        self.n_atom = args.atom

        self.pred_net = ConvNet(self.n_state, self.n_act, self.n_atom).to(self.device)

        # discrete values
        # prior knowledge of return distribution,
        self.v_min = args.v_min
        self.v_max = args.v_max

        self.v_step = ((self.v_max - self.v_min) / (self.n_atom - 1))
        self.v_range = np.linspace(self.v_min, self.v_max, self.n_atom)
        self.value_range = torch.FloatTensor(self.v_range).to(self.device)  # (N_ATOM)

        # episode step for accumulate reward
        self.epi_reward = deque(maxlen=100)

        self.step_num = args.step_num
        self.eps_greedy = 0.4 ** (1 + actor_id * 7 / (args.n_actors - 1)) \
            if args.n_actors > 1 else 0.4

        self.result = []

    def train_episode(self):
        start_time = time.time()

        # env reset
        state = np.array(self.env.reset())

        for step in range(1, self.step_num + 1):
            action = self.choose_action(state)

            # take action and get next state
            next_state, reward, done, _ = self.env.step(action)
            next_state = np.array(next_state)

            self.epi_reward.append(reward)

            # clip rewards for numerical stability
            clip_r = np.sign(reward)

            # push memory
            self.q_trace.put((state, action, clip_r, next_state, done),
                             block=True)

            # check time interval
            time_interval = round(time.time() - start_time, 2)

            # calc mean return
            mean_100_ep_return = round(np.mean([r for r in self.epi_reward]), 2)
            self.result.append(mean_100_ep_return)
            print('EPS: ', round(self.eps_greedy, 3),
                  '| Mean ep 100 return: ', mean_100_ep_return,
                  '| Used Time:', time_interval)

            state = next_state

        print("The training is done!")

    def choose_action(self, x):
        x = torch.FloatTensor(x).to(self.device).unsqueeze(0)

        if np.random.uniform() >= self.eps_greedy:
            action_value_dist = self.pred_net(x)
            action_value = torch.sum(action_value_dist * self.value_range.view(1, 1, -1), dim=2)
            action = torch.argmax(action_value, dim=1).data.cpu().numpy().item()
        else:
            action = np.random.randint(0, self.n_act)
        return action

