from collections.abc import Sized
from typing import List

import numpy as np


class ReplayMemory(Sized, object):

    __slots__ = ["_memory", "_max_size", "_count"]

    def __init__(self, max_size) -> None:
        self._count = 0
        self._memory = []
        self._max_size = max_size

    def __len__(self):
        if self._count < self._max_size:
            return self._count
        return self._max_size

    def clear(self):
        self._count = 0
        self._memory.clear()

    def insert(self, experience: List) -> None:
        # Init memory if not initilized
        if len(self._memory) == 0:
            # For each data in experience
            for e in experience:
                if isinstance(e, np.ndarray):
                    # If it is a Numpy array, preallocate memory using shape of array
                    self._memory.append(np.zeros((self._max_size, *e.shape)))
                else:
                    # Else, preallocate memory using _max_size and object type
                    self._memory.append(np.empty((self._max_size), dtype=type(e)))
        # Append experience
        for i, e in enumerate(experience):
            self._memory[i][self._count % self._max_size] = e
        # Increment _count
        self._count += 1

    def sample(self, size: int) -> List:
        # Generata random indices without replacement
        indices = np.random.choice(len(self), size)
        # Take elements from each memory using indices
        return [np.take(b, indices, axis=0) for b in self._memory]
