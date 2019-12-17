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
        self.ob_space = self.env.observation_space.shape[0]
        self.ac_space = self.env.action_space.n

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ## パラメータ
        self.batch_size = opt.batch_size
        self.roll_out_n_steps = opt.roll_out_n_steps
        self.gamma = opt.gamma

        self.eps_greedy = 0.4 ** (1 + actor_id * 7 / (opt.n_actors - 1)) \
            if opt.n_actors > 1 else 0.4

        self.n_steps = 0
        self.n_episodes = 0

        self.actor = ActorNetwork(self.ob_space, self.ac_space).to(self.device)  # ActorNetwork
        self.critic = CriticNetwork(self.ob_space).to(self.device)  # CriticNetwork

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
                actions.append(action)

                if done:
                    reward = -10

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
                final_action = self.action(final_state)
                final_value = self.value(final_state, final_action)  # Q(s,a)を出力
                self.episode_done = False

            ## discount_rewards
            rewards = self._discount_reward(rewards, final_value)  # 報酬でなく，行動価値とすることで先までの価値を知ることができる
            self.n_steps += 1

            self.q_trace.put((states, actions, rewards),
                             block=True)

    def _softmax_action(self, state):
        # state_var = to_tensor_var([state], self.use_cuda)
        state_var = torch.FloatTensor([state]).to(self.device)
        softmax_action_var = torch.exp(self.actor(state_var))  # expをかけて，行動確率とする
        softmax_action = softmax_action_var.cpu().detach().numpy()
        return softmax_action

    def exploration_action(self, state):
        softmax_action = self._softmax_action(state)

        if np.random.rand() > self.eps_greedy:
            return np.argmax(softmax_action)
        else:
            return np.random.choice(2)

    # choose an action based on state for execution
    def action(self, state):
        softmax_action = self._softmax_action(state)
        action = np.argmax(softmax_action)
        return action

    def value(self, state, action):  # Qを出力
        # state_var = to_tensor_var([state], self.use_cuda)
        state_var = torch.FloatTensor([state]).to(self.device)
        onehot_action = index2onehot(action, self.ac_space)  # 行動をonehot化
        # action_var = to_tensor_var([onehot_action], self.use_cuda)
        action_var = torch.FloatTensor([onehot_action]).to(self.device)
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



