import os
import yaml
import argparse
from datetime import datetime
import gym


from agent.iqn_agent import IQNAgent
from env import FrozenLakeEnv


def run(args):
    with open(os.path.join('config', 'iqn.yaml')) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = FrozenLakeEnv(is_slippery=True, prob=args.prob)
    test_env = FrozenLakeEnv(is_slippery=True, prob=args.prob)

    # Specify the directory to log.
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'IQN-{args.seed}-{time}')

    # Create the agent and run.
    agent = IQNAgent(
        env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
        cuda=args.cuda, **config)
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='FrozenLake-v0')
    parser.add_argument('--prob', type=float, default=1)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args)
