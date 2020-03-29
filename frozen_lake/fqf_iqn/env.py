import numpy as np
import torch

from gym.envs.toy_text import discrete

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFFF",
        "HFFF",
        "HFFF",
        "GFFF"
    ],
    "5x5": [
        "SFFF",
        "HFFF",
        "HFFF",
        "HFFF",
        "GFFF"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
    "12x4": [
        "SFFF",
        "HFFF",
        "HFFF",
        "HFFF",
        "HFFF",
        "HFFF",
        "HFFF",
        "HFFF",
        "HFFF",
        "HFFF",
        "HFFF",
        "GFFF",
    ],
    "12x6": [
        "SFFFFF",
        "HFFFFF",
        "HFFFFF",
        "HFFFFF",
        "HFFFFF",
        "HFFFFF",
        "HFFFFF",
        "HFFFFF",
        "HFFFFF",
        "HFFFFF",
        "HFFFFF",
        "GFFFFF",
    ],
}


def categorical_sample(prob_n):
    prob_n = np.asarray(prob_n)
    i = np.random.choice(list(range(len(prob_n))),
                         size=1,
                         p=prob_n/prob_n.sum())[0]
    return i


class FrozenLakeEnv(discrete.DiscreteEnv):

    def __init__(self, map_name="12x6", is_slippery=True, prob=1.0, cuda=True):
        desc = MAPS[map_name]

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_max = 100
        self.reward_min = -100
        self.reward_sub = 50
        self.step_reward = -8

        nA = 3
        self.num_actions = nA
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col-1, 0)
            elif a == DOWN:
                row = min(row+1, nrow-1)
            elif a == RIGHT:
                col = min(col+1, ncol-1)
            elif a == UP:
                row = max(row-1, 0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)

                for a in range(nA):
                    li = P[s][a]   # 参照渡し
                    letter = desc[row, col]
                    if (letter == b'G') | (letter == b'H'):
                        rew, done = self.reward_done(letter)

                        li.append((1.0, s, rew, done))
                    else:
                        if (not is_slippery) | (row == 0) | (row == self.nrow - 1):
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]

                            rew, done = self.reward_done(newletter)

                            li.append((1.0, newstate, rew, done))

                        else:
                            for b in [a, (a+1) % nA, (a+2) % nA]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]

                                rew, done = self.reward_done(newletter)

                                # 行動確率
                                if b == a:
                                    if b == 0:   # 選択行動が左なら確率1
                                        li.append((1, newstate, rew, done))
                                    else:
                                        li.append((np.round(prob, 2), newstate, rew, done))
                                elif b == 0:   # 選択をしていない左は(1-prob)の確率
                                    li.append((np.round((1-prob), 2), newstate, rew, done))
                                else:
                                    li.append((0, newstate, rew, done))

        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)

    def reset(self):
        self.s = categorical_sample(self.isd)
        self.lastaction = None
        # return torch.eye(self.nrow * self.ncol, device=self.device)[self.s]
        return self.s

    def step(self, a):
        # transitions = self.P[self.s.argmax().item()][a]
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions])
        p, s, r, d = transitions[i]
        # self.s = torch.eye(self.nrow * self.ncol, device=self.device)[s]
        self.s = s
        self.lastaction = a
        return (self.s, r, d, {"prob": p})

    def reward_done(self, newletter):
        # 報酬、終了条件作成
        if newletter == b'G':
            rew = self.reward_max
            done = True
        elif newletter == b'H':
            rew = self.reward_min
            done = True
        elif newletter == b'g':  # sub goal
            rew = self.reward_sub
            done = False
        else:
            rew = self.step_reward
            done = False

        return rew, done
