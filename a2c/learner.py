import numpy as np
import torch
from model import ActorNetwork, CriticNetwork
import gym
import torch.nn as nn
import torch.optim as optim
from utils import index2onehot, entropy

class Learner:
    def __init__(self, opt, q_batch):
        self.opt = opt
        self.q_batch = q_batch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = gym.make(self.opt.env)
        self.env.seed(self.opt.seed)
        self.n_state = self.env.observation_space.shape[0]
        self.n_act = self.env.action_space.n

        self.actor = ActorNetwork(self.n_state, self.n_act).to(self.device)
        self.critic = CriticNetwork(self.n_state).to(self.device)
        self.actor.share_memory()
        self.critic.share_memory()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=opt.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=opt.lr)

    def learning(self):
        torch.manual_seed(self.opt.seed)

        while True:
            # batch-trace
            states, actions, rewards = self.q_batch.get(block=True)

            onehot_actions = torch.FloatTensor(index2onehot(actions, self.n_act)).to(self.device)

            # update actor network
            self.actor_optimizer.zero_grad()
            action_log_probs = self.actor(states)
            action_log_probs = torch.sum(action_log_probs * onehot_actions, 1)
            values = self.critic(states)
            advantages = rewards - values.detach()
            pg_loss = -torch.sum(action_log_probs * advantages)
            actor_loss = pg_loss
            actor_loss.backward()
            self.actor_optimizer.step()

            # update critic network
            self.critic_optimizer.zero_grad()
            target_values = rewards
            critic_loss = nn.MSELoss()(values, target_values)
            critic_loss.backward()
            self.critic_optimizer.step()

            # print(' '*20, 'actor loss:{:.3f}  critic loss:{:.3f}'.format(actor_loss.item(), critic_loss.item()))
