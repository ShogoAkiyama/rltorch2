import numpy as np
import torch
from torch.optim import Adam
from model import ActorCritic
from env import make_pytorch_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Learner(object):
    def __init__(self, opt, q_batch):
        self.opt = opt
        self.q_batch = q_batch

        self.env = make_pytorch_env(self.opt.env)
        self.env.seed(self.opt.seed)
        self.n_state = self.env.observation_space.shape
        self.n_act = self.env.action_space.n

        self.net = ActorCritic(self.n_state[0], self.n_act).to(device)
        self.optimizer = Adam(self.net.parameters(), lr=opt.lr)
        self.net.share_memory()
        # self.shared_weights = shared_weights

    def learning(self):
        torch.manual_seed(self.opt.seed)
        coef_hat = torch.Tensor([[self.opt.coef_hat]]).to(device)
        rho_hat = torch.Tensor([[self.opt.rho_hat]]).to(device)
        while True:
            # self.save_model()
            # batch-trace
            state, action, reward, prob = self.q_batch.get(block=True)

            v, coef, rho, entropies, log_probs = [], [], [], [], []

            self.net.reset_recc(batch_size=self.opt.batch_size)
            for step in range(state.size(1)):
                # value[batch]
                # logit[batch, 8]
                value, logit = self.net(state[:, step, ...])
                v.append(value)
                if step >= self.opt.n_step:
                    break

                # Actorの方策: prob, Learnerの方策: logit
                action_log_prob = prob[:, step, :]
                logit_log_prob = logit.detach()

                action_prob = torch.exp(action_log_prob)
                logit_prob = torch.exp(logit_log_prob)

                is_rate = torch.cumprod(logit_prob / (action_prob + 1e-6), dim=1)[:, -1]

                # c_i: min(c, π/μ)∂
                # rho_t:
                coef.append(torch.min(coef_hat, is_rate).view(-1, 1))
                rho.append(torch.min(rho_hat, is_rate).view(-1, 1))

                # entropy
                enpy_aspace = - torch.exp(logit_log_prob) * logit_log_prob
                enpy = (enpy_aspace).sum(dim=1)
                entropies.append(enpy)

                log_probs.append(logit_log_prob)

            ####################
            # calculating loss #
            ####################
            policy_grads = 0
            v_trace = torch.zeros((state.size(1), state.size(0), 1)).to(device)
            for rev_step in reversed(range(state.size(1) - 1)):
                # value_loss[batch] v_traceの計算をするところ
                delta_v = rho[rev_step] * (
                            reward[:, rev_step].view(-1, 1) + self.opt.gamma * v[rev_step+1] - v[rev_step])

                # compute r_s + γ*v_(s+1) - V(x)  value_lossの部分はいらない気がする
                advantages = reward[:, rev_step].view(-1, 1) + self.opt.gamma * v[rev_step+1] - v[rev_step]

                v_trace[rev_step] = v[rev_step] + delta_v + self.opt.gamma * coef[rev_step] * (v_trace[rev_step+1] - v[rev_step+1])

                # 最大化
                policy_grads += rho[rev_step] * log_probs[rev_step] * advantages.detach()

            self.optimizer.zero_grad()
            value_loss = torch.sum(0.5*(v_trace - torch.stack(v))**2)
            policy_grads = policy_grads.sum()
            loss = - policy_grads.sum() \
                   + self.opt.value_loss_coef * value_loss \
                   - self.opt.entropy_coef * torch.mean(torch.stack(entropies))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.opt.max_grad_norm)
            print("value_loss {:.3f}   policy_grads {:.3f} loss {:.3f}"
                    .format(value_loss.item(), policy_grads.item(), loss.item())) 
            self.optimizer.step()
