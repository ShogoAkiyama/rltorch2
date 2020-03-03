import sys
from contextlib import closing

import numpy as np
from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete

import torch

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


def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0,0))
        while frontier:
            r, c = frontier.pop()
            if not (r,c) in discovered:
                discovered.add((r,c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if (res[r_new][c_new] not in '#H'):
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]


def categorical_sample(prob_n):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    i = np.random.choice(list(range(len(prob_n))),
                         size=1,
                         p=prob_n/prob_n.sum())[0]
    return i


class FrozenLakeEnv(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following
        SFFF
        FHFH
        FFFH
        HFFG
    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located
    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="12x6", is_slippery=True, prob=1.0, cuda=True):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_max = 100
        self.reward_min = -100
        self.step_reward = -5

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col-1,0)
            elif a == DOWN:
                row = min(row+1,nrow-1)
            elif a == RIGHT:
                col = min(col+1,ncol-1)
            elif a == UP:
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)

                for a in range(4):
                    li = P[s][a]   # 参照渡し
                    letter = desc[row, col]
                    if letter == b'G':
                        rew = self.reward_max
                        done = True

                        li.append((1.0, s, rew, done))
                    elif letter == b'H':
                        rew = self.reward_min
                        done = True

                        li.append((1.0, s, rew, done))
                    else:
                        if is_slippery:
                            for b in [a, (a+1) % 4, (a+2) % 4, (a+3) % 4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]

                                if newletter == b'G':
                                    rew = self.reward_max
                                    done = True   # torch.ByteTensor(1).to(self.device)
                                elif newletter == b'H':
                                    rew = self.reward_min
                                    done = True   # torch.ByteTensor([1]).to(self.device)
                                else:
                                    rew = self.step_reward
                                    done = False   # torch.ByteTensor([0]).to(self.device)
                                if b == a:   # 行動確率
                                    if b == 0:
                                        li.append((1, newstate, rew, done))
                                    else:
                                        li.append((prob, newstate, rew, done))
                                elif b == 0:   # 左の風
                                    li.append((1-prob, newstate, rew, done))
                                else:
                                    li.append((0, newstate, rew, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]

                            if newletter == b'G':
                                rew = self.reward_max
                                done = True
                            elif newletter == b'H':
                                rew = self.reward_min
                                done = True
                            else:
                                rew = self.step_reward
                                done = False
                            li.append((1.0, newstate, rew, done))

        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)

    def reset(self):
        self.s = categorical_sample(self.isd)
        self.lastaction = None
        return torch.eye(self.nrow * self.ncol, device=self.device)[self.s]

    def step(self, a):
        transitions = self.P[self.s.argmax().item()][a]
        i = categorical_sample([t[0] for t in transitions])
        p, s, r, d = transitions[i]
        self.s = torch.eye(self.nrow * self.ncol, device=self.device)[s]
        self.lastaction = a
        return (self.s, r, d, {"prob": p})

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            # print action
            print("({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            print("\n")
        # print map
        print("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()