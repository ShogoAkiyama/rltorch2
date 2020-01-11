import argparse
import torch.multiprocessing as mp
from actor import Actor
from learner import Learner
from memory import Memory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Some settings of the experiment.')
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--n_actors', type=int, default=3)
    parser.add_argument('--quant', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1.0e-4)
    parser.add_argument('--step_num', type=int, default=int(1e8))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.99)

    args = parser.parse_args()

    try:
        mp.set_start_method('forkserver', force=True)
        print("forkserver init")
    except RuntimeError:
        pass

    mp_manager = mp.Manager()

    processes = []

    # data communication
    q_trace = mp.Queue(maxsize=1000)
    q_batch = mp.Queue(maxsize=3)

    # QManager
    q_manager = Memory(args, q_trace, q_batch)
    p = mp.Process(target=q_manager.listening)
    p.start()
    processes.append(p)

    learner = Learner(args, q_batch)

    actors = []
    for actor_id in range(args.n_actors):
        actors.append(Actor(args, actor_id, q_trace, learner))

    for rank, actor in enumerate(actors):
        p = mp.Process(target=actor.performing)
        p.start()
        processes.append(p)

    learner.learn()
    for p in processes:
        p.join()
