import torch
import numpy as np
from network import Network
import gym
from env import make_pytorch_env


class Actor:
    def __init__(self, args, actor_id, memory, learner):
        self.actor_id = actor_id
        self.memory = memory
        self.learner = learner

        self.num_quantiles = args.quantiles
        self.cumulative_density = torch.tensor((2 * np.arange(self.num_quantiles) + 1) / (2.0 * self.num_quantiles),
                                               device=args.device, dtype=torch.float)
        self.quantile_weight = 1.0 / self.num_quantiles

        # DQN
        self.device = args.device

        self.env = make_pytorch_env()

        self.num_feats = self.env.observation_space
        self.num_actions = self.env.action_space

        # epsilon
        self.eps_greedy = 0.4 ** (1 + actor_id * 7 / (args.n_actors - 1)) \
            if args.n_actors > 1 else 0.4

        self.model = Network(self.num_feats, self.num_actions, quantiles=self.num_quantiles).to(self.device)

        self.env_state = None
        self.n_episodes = 0
        self.n_steps = 0

    def performing(self):
        while True:
            self.load_model()
            self.train_episode()

    def train_episode(self):
        self.n_episodes += 1
        done = False
        self.env_state = self.env.reset()

        while not done:
            self.n_steps += 1
            action = self.get_action(self.env_state)
            next_state, reward, done, _ = self.env.step(action)

            if not done:
                self.memory.put((self.env_state, action, reward, next_state, done), block=True)
            else:
                pass

            self.env_state = next_state

            if done:
                print(' '*20, 'Actor:', self.actor_id,
                      'Episode:', self.n_episodes, ' steps:', self.n_steps)
                self.n_steps = 0

    def get_action(self, state):
        with torch.no_grad():
            if np.random.random() > self.eps_greedy:
                action = self.action(state)
            else:
                action = np.random.randint(0, self.num_actions)
            return action

    def action(self, state):
        state = torch.FloatTensor([state]).to(self.device)
        action = (self.model(state) * self.quantile_weight).sum(dim=2).max(dim=1)[1]
        return action.item()

    def load_model(self):
        try:
            self.model.load_state_dict(self.learner.model.state_dict())
        except:
            print('load error')
