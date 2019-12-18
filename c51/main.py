import argparse
import torch.multiprocessing as mp
from actor import Actor
from learner import Learner
from memory import Memory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Some settings of the experiment.')
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--n_actors', type=int, default=1)
    parser.add_argument('--memory_capacity', type=int, default=100000)
    parser.add_argument('--atom', type=int, default=51)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--v_min', type=int, default=-5)
    parser.add_argument('--v_max', type=int, default=10)
    parser.add_argument('--step_num', type=int, default=int(1e8))
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--learn_start', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.99)

    args = parser.parse_args()

    # data communication
    q_trace = mp.Queue(maxsize=300)
    q_batch = mp.Queue(maxsize=3)

    # QManager
    q_manager = Memory(args, q_trace, q_batch)
    p = mp.Process(target=q_manager.listening)
    p.start()

    learner = Learner(args)

    actors = []
    for actor_id in range(args.n_actors):
        actors.append(Actor(args, actor_id, q_trace, learner))

    processes = []
    for rank, actor in enumerate(actors):
        p = mp.Process(target=actor.train_episode)
        p.start()
        processes.append(p)

    learner.learn()
    for p in processes:
        p.join()
