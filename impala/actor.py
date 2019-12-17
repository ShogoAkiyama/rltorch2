import torch
from model import ActorNetwork, CriticNetwork
import numpy as np
import gym

class Actor(object):
    def __init__(self, opt, actor_id,  q_trace, learner):
        self.opt = opt
        self.q_trace = q_trace
        self.learner = learner
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = gym.make(self.opt.env)
        self.env.seed(self.opt.seed + actor_id)
        self.n_state = self.env.observation_space.shape[0]
        self.n_act = self.env.action_space.n

        self.n_episodes = 0
        self.n_steps = 0
        self.gamma = opt.gamma

        # epsilon
        self.eps_greedy = 0.4 ** (1 + actor_id * 7 / (opt.n_actors - 1)) \
            if opt.n_actors > 1 else 0.4

        # モデル
        self.actor = ActorNetwork(self.n_state, self.n_act).to(self.device)
        self.critic = CriticNetwork(self.n_state).to(self.device)

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

    def _softmax_action(self, state):
        state = torch.FloatTensor([state]).to(self.device)
        softmax_action = torch.exp(self.actor(state))  # expをかけて，行動確率とする
        softmax_action = softmax_action.cpu().detach().numpy()
        return softmax_action

    def exploration_action(self, state):
        softmax_action = self._softmax_action(state)

        if np.random.rand() > self.eps_greedy:
            return np.argmax(softmax_action)
        else:
            return np.random.choice(self.n_act)

    def train_episode(self):
        done = False
        state = self.env.reset()
        self.env_state = state
        self.next_done = done

        while not done:
            self.n_steps += 1
            states = np.zeros((self.opt.n_step, self.n_state))
            actions = np.zeros(self.opt.n_step)
            rewards = np.zeros(self.opt.n_step)
            log_probs = np.zeros((self.opt.n_step, self.n_act))
            dones = np.ones(self.opt.n_step)
            for i in range(self.opt.n_step):
                states[i] = self.env_state
                dones[i] = self.next_done
                log_prob = self.actor(torch.FloatTensor([state]).to(self.device)).detach().cpu().numpy()[0]
                action = self.exploration_action(state)
                next_state, reward, done, info = self.env.step(action)

                reward = 0
                if done:
                    if self.n_steps > 190:
                        reward = 1
                    else:
                        reward = -1

                log_probs[i] = log_prob
                actions[i] = action
                rewards[i] = reward
                self.env_state = next_state
                self.next_done = done
                if done:
                    self.env_state = self.env.reset()
                    break

            # n_step回終了
            if done:
                self.n_steps = 0
                self.n_episodes += 1
                self.episode_done = True
            else:
                self.episode_done = False

            self.q_trace.put((states, actions, rewards, dones, log_probs),
                             block=True)

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
            self.actor.load_state_dict(self.learner.actor.state_dict())
            self.critic.load_state_dict(self.learner.critic.state_dict())
        except:
            print('load error')
