import torch
import numpy as np
from network import Network
from env import make_pytorch_env
import argparse
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_DIR = './model'

class Actor:
    def __init__(self, args):
        self.num_quantiles = args.quantiles
        self.cumulative_density = torch.tensor((2 * np.arange(self.num_quantiles) + 1) / (2.0 * self.num_quantiles),
                                               device=args.device, dtype=torch.float)
        self.quantile_weight = 1.0 / self.num_quantiles

        self.device = args.device

        self.env = make_pytorch_env()

        self.num_feats = self.env.observation_space
        self.num_actions = self.env.action_space

        self.model = Network(self.num_feats, self.num_actions, quantiles=self.num_quantiles).to(self.device)

        model_path = sorted(glob(os.path.join(MODEL_DIR, '*')))[-1]
        self.model.load_state_dict(torch.load(model_path))

    def performing(self):
        rewards = []

        state = self.env.reset()
        action = self.action(state)
        state, reward, done, _ = self.env.step(action)
        rewards.append(reward)

        while not done:
            action = self.action(state)
            state, reward, done, _ = self.env.step(action)
            rewards.append(reward)

        rewards_mu = np.array([np.sum(np.array(l_i), 0) for l_i in rewards]).sum()
        print('Eval Reward %.2f' % (rewards_mu))

    def action(self, state):
        state = torch.FloatTensor([state]).to(self.device)
        action = (self.model(state) * self.quantile_weight)

        dist_action = action[0].cpu().detach().numpy()
        sns.distplot(dist_action[0], bins=10, color='red')
        sns.distplot(dist_action[1], bins=10, color='green')
        plt.show()

        action = action.sum(dim=2).max(dim=1)[1]
        return action.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Some settings of the experiment.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--device', type=str, default=device)

    # quantile
    parser.add_argument('--quantiles', type=int, default=51)

    args = parser.parse_args()

    actor = Actor(args)

    actor.performing()
