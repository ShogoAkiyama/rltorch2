import argparse
import torch.multiprocessing as mp
import torch

from qmaneger import QManeger
from actor import Actor
from learner import Learner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Some settings of the experiment.')
    parser.add_argument('--env_id', type=str, default="CartPole-v0")
    parser.add_argument('-n', '--n_actors', type=int, default=3)
    parser.add_argument('--update_freq', type=int, default=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--device', type=str, default=device)

    # misc
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--learning_rate', type=float, default=2.5e-4)

    # memory
    parser.add_argument('--target_net_update_freq', type=int, default=1000)
    # parser.add_argument('--exp_replay_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)

    # learning control
    parser.add_argument('--learn_start', type=int, default=10000)
    # parser.add_argument('--max_frames', type=int, default=1000000)

    # Nstep
    parser.add_argument('--multi_steps', type=int, default=1)

    # quantile
    parser.add_argument('--quantiles', type=int, default=51)

    args = parser.parse_args()

    try:
        mp.set_start_method('forkserver', force=True)
        print("forkserver init")
    except RuntimeError:
        pass

    mp_manager = mp.Manager()

    processes = []

    # data communication
    memory = mp.Queue(maxsize=300)
    q_batch = mp.Queue(maxsize=3)

    # QManager
    q_manager = QManeger(args, memory, q_batch)
    p = mp.Process(target=q_manager.listening)
    p.start()

    processes.append(p)

    learner = Learner(args, q_batch)  # inner shared network was used by actors.

    actors = []
    for actor_id in range(args.n_actors):
        actors.append(Actor(args, actor_id, memory, learner))

    for rank, actor in enumerate(actors):
        p = mp.Process(target=actor.performing)
        p.start()
        processes.append(p)

    learner.learn()
    for p in processes:
        p.join()
