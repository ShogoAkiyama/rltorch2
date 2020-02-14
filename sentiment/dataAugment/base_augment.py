import random
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool

# from utils.method import Method
# from utils.action import Action

class Augmenter:
    def __init__(self, method, action, aug_min, aug_max, aug_p=0.1, device='cpu'):
        # self.name = name
        self.action = action
        self.method = method
        self.aug_min = aug_min
        self.aug_max = aug_max
        self.aug_p = aug_p
        self.device = device

    def augment(self, data, n=1, num_thread=1):
        max_retry_times = 1  # max loop times of n to generate expected number of outputs

        results = []
        action_fx = None
<<<<<<< HEAD
        clean_data = self.clean(data)
        if self.action == 'insert':
=======
        # clean_data = self.clean(data)
        clean_data = data
        if self.action == Action.INSERT:
>>>>>>> 488c34e86c712194a41de1667da21993c8a3d36a
            action_fx = self.insert
        elif self.action == 'substitute':
            action_fx = self.substitute
        elif self.action == 'swap':
            action_fx = self.swap
        elif self.action == 'delete':
            action_fx = self.delete
        elif self.action == 'split':
            action_fx = self.split

        for _ in range(max_retry_times+1):
            augmented_results = []
            if num_thread == 1 or self.device == 'cuda':
                # TODO: support multiprocessing for GPU
                # https://discuss.pytorch.org/t/using-cuda-multiprocessing-with-single-gpu/7300
                augmented_results = [action_fx(clean_data) for _ in range(n)]
            else:
                augmented_results = self._parallel_augment(action_fx, clean_data, n=n, num_thread=num_thread)

            for augmented_result in augmented_results:
                if not self.is_duplicate(results + [data], augmented_result):
                    results.append(augmented_result)

                if len(results) >= n:
                    break
            if len(results) >= n:
                break

        # TODO: standardize output to list even though n=1 from 1.0.0
        if len(results) == 0:
            # if not result, return itself
            if n == 1:
                return data
            else:
                return [data]
        if n == 1:
            return results[0]
        return results[:n]

    def augments(self, data, n=1, num_thread=1):
        augmented_results = []
        if num_thread == 1 or self.device == 'cuda':
            for d in data:
                augmented_result = self.augment(data=d, n=n, num_thread=1)  # TOOD: cuda does not support mulithread
                if n == 1:
                    augmented_results.append(augmented_result)
                else:
                    augmented_results.extend(augmented_result)
        else:
            batch_data = [data[i:i+num_thread] for i in range(0, len(data), num_thread)]
            for i in range(n):
                for mini_batch_data in batch_data:
                    augmented_results.extend(self._parallel_augments(self.augment, mini_batch_data))

        return augmented_results

    @classmethod
    def _parallel_augment(cls, action_fx, data, n, num_thread=2):
        pool = ThreadPool(num_thread)
        results = pool.map(action_fx, [data] * n)
        pool.close()
        pool.join()
        return results

    @classmethod
    def _parallel_augments(cls, action_fx, data):
        pool = ThreadPool(len(data))
        results = pool.map(action_fx, data)
        pool.close()
        pool.join()
        return results

    @classmethod
    def is_duplicate(cls, dataset, data):
        raise NotImplementedError

    @classmethod
    def prob(cls):
        return np.random.random()

    @classmethod
    def sample(cls, x, num):
        if isinstance(x, list):
            return random.sample(x, num)
        elif isinstance(x, int):
            return np.random.randint(1, x-1)

    @classmethod
    def clean(cls, data):
        raise NotImplementedError

    def generate_aug_cnt(self, size, aug_p=None):
        if aug_p is not None:
               percent = aug_p
        elif self.aug_p is not None:
            percent = self.aug_p
        else:
            percent = 0.3
        cnt = int(percent * size)

        if cnt < self.aug_min:
            return self.aug_min
        if self.aug_max is not None and cnt > self.aug_max:
            return self.aug_max
        return cnt

    def generate_aug_idxes(self, inputs):
        aug_cnt = self.generate_aug_cnt(len(inputs))
        token_idxes = [i for i, _ in enumerate(inputs)]
        aug_idxes = self.sample(token_idxes, aug_cnt)
        return aug_idxes

    # def __str__(self):
    #     return 'Name:{}, Action:{}, Method:{}'.format(self.name, self.action, self.method)
