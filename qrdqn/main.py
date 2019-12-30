import gym


import torch

from timeit import default_timer as timer
import math
from agents.QRDQN import QRDQN


class Config(object):
    def __init__(self):
        pass

config = Config()
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config.UPDATE_FREQ = 1
config.ACTION_SELECTION_COUNT_FREQUENCY = 1000

# epsilon variables
config.epsilon_start = 1.0
config.epsilon_final = 0.01
config.epsilon_decay = 60000
config.epsilon_by_frame = lambda frame_idx: config.epsilon_final + (
            config.epsilon_start - config.epsilon_final) * math.exp(-1. * frame_idx / config.epsilon_decay)

# misc agent variables
config.GAMMA = 0.99
config.LR = 2.5e-4

# memory
config.TARGET_NET_UPDATE_FREQ = 1000
config.EXP_REPLAY_SIZE = 100000
config.BATCH_SIZE = 32

# Learning control variables
config.LEARN_START = 10000
config.MAX_FRAMES = 1000000

# Nstep controls
config.N_STEPS = 1

# Quantile Regression Parameters
config.QUANTILES = 51




start = timer()

env_id = "CartPole-v0"
env = gym.make(env_id)
model = QRDQN(env=env, config=config)

episode_reward = 0

observation = env.reset()
for frame_idx in range(1, config.MAX_FRAMES + 1):
    epsilon = config.epsilon_by_frame(frame_idx)

    action = model.get_action(observation, epsilon)
    prev_observation = observation
    observation, reward, done, _ = env.step(action)
    observation = None if done else observation

    model.update(prev_observation, action, reward, observation, frame_idx)
    episode_reward += reward

    if done:
        model.finish_nstep()
        observation = env.reset()
        model.save_reward(episode_reward)
        print('episode_reward: ', episode_reward)
        episode_reward = 0

model.save_w()
env.close()
