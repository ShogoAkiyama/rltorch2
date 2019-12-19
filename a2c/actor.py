import numpy as np
import gym

# torchのライブラリ
import torch
import torch.nn as nn
import torch.optim as optim
from model import ActorNetwork, CriticNetwork
from qmaneger import QManeger
from utils import index2onehot, entropy


class Actor:
    def __init__(self, opt, actor_id,  q_trace, learner):
        self.opt = opt
        self.q_trace = q_trace
        self.learner = learner

        self.env = gym.make(self.opt.env)
        self.env_state = self.env.reset()
        self.n_state = self.env.observation_space.shape[0]
        self.n_act = self.env.action_space.n

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ## パラメータ
        self.batch_size = opt.batch_size
        self.roll_out_n_steps = opt.roll_out_n_steps
        self.gamma = opt.gamma

        self.eps_greedy = 0.4 ** (1 + actor_id * 7 / (opt.n_actors - 1)) \
            if opt.n_actors > 1 else 0.4

        self.n_episodes = 0

        self.actor = ActorNetwork(self.n_state, self.n_act).to(self.device)  # ActorNetwork
        self.critic = CriticNetwork(self.n_state).to(self.device)  # CriticNetwork

    def performing(self):
        torch.manual_seed(self.opt.seed)

        while True:
            self.load_model()
            self.train_episode()
            if self.n_episodes % 100 == 0:
                rewards = self.evaluation(self.env)
                rewards_mu = np.array([np.sum(np.array(l_i), 0) for l_i in rewards]).mean()
                print("Episode %d, Average Reward %.2f"
                      % (self.n_episodes, rewards_mu))

    def train_episode(self):
        done = False
        state = self.env.reset()
        self.env_state = state

        while not done:
            states = []
            actions = []
            rewards = []
            for i in range(self.roll_out_n_steps):
                states.append(self.env_state)
                action = self.exploration_action(self.env_state)
                next_state, reward, done, _ = self.env.step(action)

                # reward = 0
                if done:
                    reward = -10
                    # if self.n_steps > 190:
                    #     reward = 1
                    # else:
                    #     reward = -1

                actions.append(action)
                rewards.append(reward)
                self.env_state = next_state
                final_state = next_state
                if done:
                    self.env_state = self.env.reset()
                    break

            # n_step回終了
            if done:
                final_value = 0.0
                self.n_episodes += 1
                self.episode_done = True
            else:
                final_value = self.value(final_state)
                self.episode_done = False

            ## discount_rewards
            rewards = self._discount_reward(rewards, final_value)

            self.q_trace.put((states, actions, rewards),
                             block=True)

    def _softmax_action(self, state):
        state = torch.FloatTensor([state]).to(self.device)
        softmax_action = torch.exp(self.actor(state))  # expをかけて，行動確率とする
        softmax_action = softmax_action.cpu().detach().numpy()
        return softmax_action

    def exploration_action(self, state):
        softmax_action = self._softmax_action(state)

        if np.random.rand() > self.eps_greedy:
            action = np.argmax(softmax_action)
        else:
            action = np.random.choice(self.n_act)
        return action

    # choose an action based on state for execution
    def action(self, state):
        softmax_action = self._softmax_action(state)
        action = np.argmax(softmax_action)
        return action

    def value(self, state):  # Qを出力
        state_var = torch.FloatTensor([state]).to(self.device)
        q_var = self.critic(state_var)  # 行動価値を出value
        q = q_var.cpu().detach().numpy()
        return q

    def _discount_reward(self, rewards, final_value):
        discounted_r = np.zeros_like(rewards)
        R = final_value  # Q(s_t, a_t)
        for t in reversed(range(0, len(rewards))):
            R = rewards[t] + self.gamma * R
            discounted_r[t] = R
        return discounted_r

    def evaluation(self, env_eval):
        rewards = []
        for i in range(10):
            rewards_i = []

            state = env_eval.reset()
            action = self.action(state)
            state, reward, done, _ = env_eval.step(action)
            rewards_i.append(reward)

            while not done:
                action = self.action(state)
                state, reward, done, _ = env_eval.step(action)
                rewards_i.append(reward)
            rewards.append(rewards_i)

        return rewards

    def load_model(self):
        try:
            # self.net.load_state_dict(self.learner.net.state_dict())
            self.actor.load_state_dict(self.learner.actor.state_dict())
            self.critic.load_state_dict(self.learner.critic.state_dict())
        except:
            print('load error')



