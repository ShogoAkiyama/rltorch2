import torch
from torch.optim import Adam
from model import ActorCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Learner(object):
    def __init__(self, opt, q_batch):
        self.opt = opt
        self.q_batch = q_batch
        self.network = ActorCritic(opt).to(device)
        self.optimizer = Adam(self.network.parameters(), lr=opt.lr)
        self.network.share_memory()

    def learning(self):
        torch.manual_seed(self.opt.seed)
        coef_hat = torch.Tensor([[self.opt.coef_hat]]).to(device)
        rho_hat = torch.Tensor([[self.opt.rho_hat]]).to(device)
        while True:
            # batch-trace
            # s[batch, n_step+1, 3, width, height]
            # a[batch, n_step, a_space]
            # rew[batch, n_step]
            # a_prob[batch, n_step, a_space]
            s, a, rew, prob = self.q_batch.get(block=True)
            ###########################
            # variables we need later #
            ###########################
            v, coef, rho, entropies, log_prob = [], [], [], [], []
            cx = torch.zeros(self.opt.batch_size, 256).to(device)
            hx = torch.zeros(self.opt.batch_size, 256).to(device)
            for step in range(s.size(1)):
                # value[batch]
                # logit[batch, 12]
                value, logit, (hx, cx) = self.network((s[:, step, ...], (hx, cx)))
                v.append(value)
                if step >= a.size(1):  # noted that s[, n_step+1, ...] but a[, n_step,...]
                    break              # loop for n_step+1 because v in n_step+1 is needed.

                # π/μ[batch]
                # TODO: cumprod might produce runtime problem
                logit_a = a[:, step, :] * logit.detach() + (1 - a[:, step, :]) * (1 - logit.detach())
                prob_a = a[:, step, :] * prob[:, step, :] + (1 - a[:, step, :]) * (1 - prob[:, step, :])
                is_rate = torch.cumprod(logit_a/(prob_a + 1e-6), dim=1)[:, -1]
                coef.append(torch.min(coef_hat, is_rate))
                rho.append(torch.min(rho_hat, is_rate))

                # enpy_aspace[batch, 12]
                # calculating the entropy[batch, 1]
                # more specifically there are [a_space] entropy for each batch, sum over them here.
                # noted that ~do not~ use detach here
                enpy_aspace = - torch.log(logit) * logit - torch.log(1-logit) * (1-logit)
                enpy = (enpy_aspace).sum(dim=1, keepdim=True)
                entropies.append(enpy)

                # calculating the prob that the action is taken by target policy
                # and the prob_pi_a[batch, 12] and log_prob[batch, 1] of this action
                # noted that ~do not~ use detach here
                prob_pi_a = (a[:, step, :] * logit) + (1 - a[:, step, :]) * (1 - logit)
                log_prob_pi_a = torch.log(prob_pi_a).sum(dim=1, keepdim=True)
                log_prob.append(log_prob_pi_a)
                # prob_pi_a = torch.cumprod(prob_pi_a, dim=1)[:, -1:]
                # log_prob_pi_a = torch.log(prob_pi_a)

            ####################
            # calculating loss #
            ####################
            policy_loss = 0
            value_loss = 0
            # gae = torch.zeros(self.opt.batch_size, 1)
            for rev_step in reversed(range(s.size(1) - 1)):
                # compute v_(s+1)[batch] for policy gradient
                fix_vp = rew[:, rev_step] + self.opt.gamma * (v[rev_step+1] + value_loss) - v[rev_step]

                # value_loss[batch]
                td = rew[:, rev_step] + self.opt.gamma * v[rev_step + 1] - v[rev_step]
                value_loss = self.opt.gamma * coef[rev_step] * value_loss + rho[rev_step] * td

                # policy_loss = policy_loss - log_probs[i] * Variable(gae)
                # the td must be detach from network-v

                # # dalta_t[batch]
                # delta_t = rew[:, rev_step] + self.opt.gamma * v[rev_step + 1] - v[rev_step]
                # gae = gae * self.opt.gamma + delta_t.detach()

                policy_loss = policy_loss \
                              - rho[rev_step] * log_prob[rev_step] * fix_vp.detach() \
                              - self.opt.entropy_coef * entropies[rev_step]

            self.optimizer.zero_grad()
            policy_loss = policy_loss.sum()
            value_loss = value_loss.sum()
            loss = policy_loss + self.opt.value_loss_coef * value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.opt.max_grad_norm)
            print("v_loss {:.3f} p_loss {:.3f}".format(value_loss.item(), policy_loss.item()))
            self.optimizer.step()
