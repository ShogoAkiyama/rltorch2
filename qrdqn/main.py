import argparse
from timeit import default_timer as timer
import gym
import numpy as np
import torch
from agents.QRDQN import QRDQN


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Some settings of the experiment.')
    parser.add_argument('--env_id', type=str, default="CartPole-v0")
    parser.add_argument('--update_freq', type=int, default=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--device', type=str, default=device)

    # epsilon
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_final', type=float, default=0.01)
    parser.add_argument('--epsilon_decay', type=int, default=60000)

    # misc
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--learning_rate', type=float, default=2.5e-4)

    # memory
    parser.add_argument('--target_net_update_freq', type=int, default=1000)
    parser.add_argument('--exp_replay_size', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=32)

    # learning control
    parser.add_argument('--learn_start', type=int, default=10000)
    parser.add_argument('--max_frames', type=int, default=1000000)

    # Nstep
    parser.add_argument('--n_steps', type=int, default=1)

    # quantile
    parser.add_argument('--QUANTILES', type=int, default=51)

    args = parser.parse_args()

    # agent
    start = timer()
    model = QRDQN(args=args)
    model.train_episode()
