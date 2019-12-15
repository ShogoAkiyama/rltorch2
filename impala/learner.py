import numpy as np
import torch
from torch.optim import Adam
from model import ActorCritic
from env import make_pytorch_env

class Learner(object):
    def __init__(self, opt, q_batch):
        self.opt = opt
        self.q_batch = q_batch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = make_pytorch_env(self.opt.env)
        self.env.seed(self.opt.seed)
        self.n_state = self.env.observation_space.shape
        self.n_act = self.env.action_space.n

        self.net = ActorCritic(self.n_state[0], self.n_act).to(self.device)
        self.optimizer = Adam(self.net.parameters(), lr=opt.lr)
        self.net.share_memory()
        # self.shared_weights = shared_weights

    def learning(self):
        torch.manual_seed(self.opt.seed)
        coef_hat = torch.Tensor([[self.opt.coef_hat]]).to(self.device)
        rho_hat = torch.Tensor([[self.opt.rho_hat]]).to(self.device)
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
                onehot_actions = self.idx2onehot(action[:, step], self.n_act)

                action_log_prob = prob[:, step, :]
                logit_log_prob = logit
                action_prob = torch.exp(action_log_prob)
                logit_prob = torch.exp(logit_log_prob)

                action_log_prob = torch.sum(action_log_prob * onehot_actions, 1)
                logit_log_prob = torch.sum(logit_log_prob * onehot_actions, 1)
                action_prob = torch.sum(action_prob * onehot_actions, 1)
                logit_prob = torch.sum(logit_prob * onehot_actions, 1)

                is_rate = torch.prod(logit_prob.detach() / (action_prob.detach() + 1e-6))

                # c_i: min(c, π/μ)∂
                # rho_t:
                coef.append(torch.min(coef_hat, is_rate))
                rho.append(torch.min(rho_hat, is_rate))

                # entropy
                enpy_aspace = - torch.sum(logit_prob * logit_log_prob)
                entropies.append(enpy_aspace)

                log_probs.append(logit_log_prob)

            ####################
            # calculating loss #
            ####################
            policy_grads = 0
            v_trace = torch.zeros((state.size(1), state.size(0), 1)).to(self.device)
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
            policy_grads = -policy_grads.sum()
            value_loss = torch.sum(0.5*(v_trace.detach() - torch.stack(v))**2)
            entropy_loss = torch.mean(torch.stack(entropies))
            loss = policy_grads \
                   + self.opt.value_loss_coef * value_loss \
                #    - self.opt.entropy_coef * entropy_loss

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.opt.max_grad_norm)
            # print("value_loss:", value_loss, "   policy_grads:", policy_grads,
            #       "   loss: ", loss)
            print("value_loss {:.3f}   policy_grads {:.3f}   loss {:.3f}"
                    .format(value_loss.item(), policy_grads.item(), loss.item())) 
            self.optimizer.step()

    def idx2onehot(self, idx, dim):
        if isinstance(idx, np.int) or isinstance(idx, np.int64):
            one_hot = np.zeros(dim)
            one_hot[idx] = 1.
        else:   # indexが多次元
            one_hot = torch.eye(self.n_act)[idx.cpu().numpy().astype(np.int8)].to(self.device)
        return one_hot
