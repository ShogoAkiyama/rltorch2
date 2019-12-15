import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch.multiprocessing as mp
# from torch.multiprocessing import Queue
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (6144, rlimit[1]))
import argparse

from qmaneger import QManeger
from learner import Learner
from actor import Actor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMPALA')
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('-n', '--n_actors', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--gpu', type=int, default=0, help='gpu number')

    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-grad-norm', type=float, default=50)
    parser.add_argument('--value-loss-coef', type=float, default=0.5)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--n-step', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--coef-hat', type=float, default=1.0)
    parser.add_argument('--rho-hat', type=float, default=1.0)

    parser.add_argument('--max-episode-length', type=int, default=100000)

    args = parser.parse_args()

    try:
        mp.set_start_method('forkserver', force=True)
        print("forkserver init")
    except RuntimeError:
        pass

    # shared weights
    mp_manager = mp.Manager()
    shared_weights = mp_manager.dict()

    processes = []

    # data communication
    q_trace = mp.Queue(maxsize=300)
    q_batch = mp.Queue(maxsize=3)

    # QManager
    q_manager = QManeger(args, q_trace, q_batch)
    p = mp.Process(target=q_manager.listening)
    p.start()

    processes.append(p)
    learner = Learner(args, q_batch)  # inner shared network was used by actors.

    # actors = [Actor(args, q_trace, learner)]
    actors = []
    for actor_id in range(args.n_actors):
        actors.append(Actor(args, actor_id, q_trace, learner))

    for rank, actor in enumerate(actors):
        p = mp.Process(target=actor.performing)
        p.start()
        processes.append(p)

    learner.learning()
    for p in processes:
        p.join()
