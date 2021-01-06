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
        if not len(self._memory):
            # For each data in experience
            for e in experience:
                # If it is a Numpy array, preallocate memory using shape of array
                # else, preallocate memory using _max_size and object type
                dtype = e.dtype if isinstance(e, np.ndarray) else type(e)
                shape = (self._max_size, *e.shape) if isinstance(e, np.ndarray) else (self._max_size)
                self._memory.append(np.empty(shape, dtype))
        # Append experience
        for i, e in enumerate(experience):
            self._memory[i][self._count % self._max_size] = e
        # Increment _count
        self._count += 1

    def sample(self, size: int) -> List:
        # Generata random indices without replacement
        indices = np.random.choice(len(self), size)
        # Take elements from each memory using indices
        return [np.take(m, indices, axis=0) for m in self._memory]
