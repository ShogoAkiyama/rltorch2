import torch

def grad_false(m):
    for param in m.parameters():
        param.requires_grad = False


def weighted_square_error(input, target, weight):
    return torch.mean((input - target)**2 * weight)


def process_reward(reward, is_episode_end, done):
    if is_episode_end:
        return reward - 8.0
    else:
        return reward

