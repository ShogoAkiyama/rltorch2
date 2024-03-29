import os
import yaml
import argparse
from datetime import datetime

from fqf_iqn.agent import IQNAgent
from fqf_iqn.env import FrozenLakeEnv


def run(args):
    with open(os.path.join('fqf_iqn', 'config', 'iqn.yaml')) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = FrozenLakeEnv(is_slippery=True, prob=args.prob, cuda=args.cuda)
    test_env = FrozenLakeEnv(is_slippery=True, prob=args.prob, cuda=args.cuda)

    # Specify the directory to log.
    time = datetime.now().strftime("%Y%m%d-%H%M")
    c = args.c
    risk = ''
    if c != 0:
        if args.sensitive:
            risk = 'sensitive'
        else:
            risk = 'taking'
    log_dir = os.path.join(
        'logs', args.env_id, 'IQN', f'IQN-c{c}-{risk}-{args.seed}-{time}')

    # Create the agent and run.
    agent = IQNAgent(
        env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
        c=args.c, sensitive=args.sensitive,
        cuda=args.cuda, **config)

    if args.eval:
        agent.load_models(os.path.join(
            'logs', 'FrozenLake-v0', f'IQN-c0', 'model', 'best'))
        agent.evaluate()
    else:
        agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='FrozenLake-v0')
    parser.add_argument('--prob', type=float, default=0.9)
    parser.add_argument('--sensitive', action='store_true')
    parser.add_argument('--c', type=float, default=0)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    run(args)
