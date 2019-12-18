import argparse
import torch.multiprocessing as mp
from actor import Actor
from learner import Learner
from qmaneger import QManeger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMPALA')
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('-n', '--n_actors', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--gpu', type=int, default=0, help='gpu number')

    parser.add_argument('--lr', type=float, default=2.5e-4, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--roll_out_n_steps', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--max-episode-length', type=int, default=100000)

    args = parser.parse_args()

    try:
        mp.set_start_method('forkserver', force=True)
        print("forkserver init")
    except RuntimeError:
        pass

    # shared weights
    mp_manager = mp.Manager()

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

