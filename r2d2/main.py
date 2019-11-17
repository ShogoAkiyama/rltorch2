import sys
import os
import argparse
import shutil
import torch.multiprocessing as mp

sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from r2d2.actor import actor_process
from r2d2.learner import learner_process

mp.set_start_method('spawn', force=True)

def run():
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('-n', '--n_actors', type=int, default=5)

    parser.add_argument('-gamma', type=float, default=0.99)
    parser.add_argument('-priority_exp', type=float, default=0.6)
    parser.add_argument('-importance_exp', type=float, default=0.4)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-multi_step', type=int, default=3)
    parser.add_argument('-memory_size', type=int, default=100)
    parser.add_argument('-update_per_epoch', type=int, default=100)


    parser.add_argument('-eta', type=float, default=0.9)
    parser.add_argument('-seq_size', type=int, default=20)
    parser.add_argument('-overlap_size', type=int, default=10)
    parser.add_argument('-burn_in_size', type=int, default=10)

    parser.add_argument('-actor_save_memory', type=int, default=5)
    parser.add_argument('-actor_load_model', type=int, default=5)
    parser.add_argument('-actor_save_log', type=int, default=100)

    parser.add_argument('-learner_eval', type=int, default=10)
    parser.add_argument('-learner_load_memory', type=int, default=5)
    parser.add_argument('-learner_save_model', type=int, default=10)
    parser.add_argument('-learner_target_update', type=int, default=10)
    parser.add_argument('-learner_save_log', type=int, default=10)

    parser.add_argument('-lr', type=float, default=2.5e-4)

    args = parser.parse_args()
    shared_memory = mp.Queue(100)
    mp_manager = mp.Manager()
    shared_weights = mp_manager.dict()

    # where to log
    memory_dir = os.path.join(
        './', 'logs', 'memory')
    summary_dir = os.path.join(
        './', 'logs', 'summary')
    if not os.path.exists(memory_dir):
        os.makedirs(memory_dir)
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    else:
        shutil.rmtree(summary_dir)
        os.makedirs(summary_dir)

    # learner process
    processes = [mp.Process(
        target=learner_process,
        args=(args, shared_memory, shared_weights))]

    # processes = []
    for actor_id in range(args.n_actors):
        processes.append(mp.Process(
            target=actor_process,
            args=(actor_id, args, shared_memory, shared_weights)))

    for pi in range(len(processes)):
        processes[pi].start()

    for p in processes:
        p.join()

if __name__ == '__main__':
    run()