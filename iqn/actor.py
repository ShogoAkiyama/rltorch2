import gym
import torch
import numpy as np

from model import ConvNet


class Actor:
    def __init__(self, args, actor_id, q_trace, learner):
        self.seed = args.seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_trace = q_trace
        self.learner = learner

        self.env = gym.make(args.env)
        self.env_state = self.env.reset()
        self.n_act = self.env.action_space.n
        self.n_state = self.env.observation_space.shape[0]
        self.n_quant = args.quant

        self.net = ConvNet(self.n_state, self.n_act, self.n_quant).to(self.device)

        self.step_num = args.step_num
        self.eps_greedy = 0.4 ** (1 + actor_id * 7 / (args.n_actors - 1)) \
            if args.n_actors > 1 else 0.4

        self.n_episodes = 0
        self.n_steps = 0

    def performing(self):
        torch.manual_seed(self.seed)

        while True:
            self.load_model()
            self.train_episode()
            if self.n_episodes % 10 == 0:
                rewards = self.evaluation(self.env)
                rewards_mu = np.array([np.sum(np.array(l_i), 0) for l_i in rewards]).mean()
                print("Episode %d, Average Reward %.2f"
                      % (self.n_episodes, rewards_mu))

    def train_episode(self):
        done = False
        self.env_state = self.env.reset()

        while not done:
            self.n_steps += 1
            action = self.choose_action(self.env_state)
            next_state, reward, done, _ = self.env.step(action)

            reward = 0
            if done:
                reward = -1

            # push memory
            self.q_trace.put((self.env_state, action, reward, next_state, done),
                             block=True)

            self.env_state = next_state
            if done:
                self.env_state = self.env.reset()
                break

        if done:
            self.n_steps = 0
            self.n_episodes += 1
            self.episode_done = True
        else:
            self.episode_done = False

    def choose_action(self, state):
        if np.random.uniform() >= self.eps_greedy:
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            action_value, _ = self.net(state)
            action_value = action_value[0].sum(dim=0)
            action = torch.argmax(action_value).detach().cpu().item()
        else:
            action = np.random.randint(0, self.n_act)
        return action

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

    def action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action_value, _ = self.net(state)
        action_value = action_value[0].sum(dim=0)
        action = torch.argmax(action_value).detach().cpu().item()
        return action

    def load_model(self):
        try:
            self.net.load_state_dict(self.learner.net.state_dict())
        except:
            print('load error')
