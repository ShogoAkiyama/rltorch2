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
        self.n_atom = args.atom

        self.pred_net = ConvNet(self.n_state, self.n_act, self.n_atom).to(self.device)

        # パラメータ
        self.v_min = args.v_min
        self.v_max = args.v_max

        self.v_step = ((self.v_max - self.v_min) / (self.n_atom - 1))
        self.v_range = np.linspace(self.v_min, self.v_max, self.n_atom)
        self.value_range = torch.FloatTensor(self.v_range).to(self.device)  # (N_ATOM)

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
        state = self.env.reset()
        self.env_state = state

        while not done:
            self.n_steps += 1
            action = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)

            # clip_r = np.sign(reward)
            reward = 0
            if done:
                if self.n_steps > 190:
                    reward = 1
                else:
                    reward = -1

            # push memory
            self.q_trace.put((state, action, reward, next_state, done),
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
            action_value_dist = self.pred_net(state)
            action_value = torch.sum(action_value_dist * self.value_range.view(1, 1, -1), dim=2)
            action = torch.argmax(action_value, dim=1).data.cpu().numpy().item()
        else:
            action = np.random.randint(0, self.n_act)
        return action

    def evaluation(self, env_eval):
        rewards = []
        for i in range(2):
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
        state = torch.FloatTensor([state]).to(self.device)
        # [n_act, 51]
        action_value_dist = self.pred_net(state)
        action_value = torch.sum(action_value_dist * self.value_range.view(1, 1, -1), dim=2)
        action = torch.argmax(action_value, dim=1).data.cpu().numpy().item()
        return action

    def load_model(self):
        try:
            self.pred_net.load_state_dict(self.learner.pred_net.state_dict())
        except:
            print('load error')
