import numpy as np
import torch
from model import ActorNetwork, CriticNetwork
import gym
import torch.nn as nn
import torch.optim as optim
from utils import idx2onehot, entropy


class Learner(object):
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
        coef_hat = torch.FloatTensor([self.opt.coef_hat]*self.opt.batch_size*self.opt.n_step).view(self.opt.batch_size, self.opt.n_step)
        rho_hat = torch.FloatTensor([self.opt.rho_hat]*self.opt.batch_size*self.opt.n_step).view(self.opt.batch_size, self.opt.n_step)
        while True:
            # batch-trace
            states, actions, rewards, dones, action_log_probs = self.q_batch.get(block=True)

            logit_log_probs = self.actor(states)
            V = self.critic(states).view(self.opt.batch_size, self.opt.n_step) * (1 - dones)

            action_probs = torch.exp(action_log_probs)
            logit_probs = torch.exp(logit_log_probs)

            is_rate = torch.prod(logit_probs / (action_probs + 1e-6), dim=-1).detach()
            coef = torch.min(coef_hat, is_rate) * (1 - dones)
            rho = torch.min(rho_hat, is_rate) * (1 - dones)

            # V-trace
            v_trace = torch.zeros((self.opt.batch_size, self.opt.n_step)).to(self.device)
            target_V = V.detach()
            for rev_step in reversed(range(states.size(1) - 1)):
                v_trace[:, rev_step] = target_V[:, rev_step] \
                                       + rho[:, rev_step] * (rewards[:, rev_step] + self.opt.gamma*target_V[:, rev_step+1] - target_V[:, rev_step]) \
                                       + self.opt.gamma * coef[:, rev_step] * (v_trace[:, rev_step+1] - target_V[:, rev_step+1])

            # actor loss
            onehot_actions = torch.FloatTensor(
                idx2onehot(actions.cpu().numpy(), self.opt.batch_size, self.n_act)).to(self.device)
            logit_log_probs = torch.sum(logit_log_probs * onehot_actions, dim=-1)
            advantages = rewards + self.opt.gamma * v_trace - V
            pg_loss = -torch.sum(logit_log_probs * advantages.detach())
            actor_loss = pg_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # critic
            critic_loss = torch.mean((v_trace.detach() - V)**2)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # print('actor loss:{:.3f}  critic loss:{:.3f}'.format(actor_loss.item(), critic_loss.item()))

