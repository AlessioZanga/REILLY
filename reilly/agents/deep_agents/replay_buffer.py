from collections.abc import Sized
from typing import List

import numpy as np


class ReplayBuffer(Sized, object):

    __slots__ = ["_buffer", "_max_size", "_count"]

    def __init__(self, max_size) -> None:
        self._count = 0
        self._buffer = []
        self._max_size = max_size
    
    def __len__(self):
        if self._count < self._max_size:
            return self._count
        return self._max_size
    
    def clear(self):
        self._count = 0
        self._buffer.clear()

    def insert(self, experience: List) -> None:
        # Init buffer if not initilized
        if len(self._buffer) == 0:
            # For each data in experience
            for e in experience:
                if isinstance(e, np.ndarray):
                    # If it is a Numpy array, preallocate buffer using shape of array
                    self._buffer.append(np.zeros((self._max_size, *e.shape)))
                else:
                    # Else, preallocate buffer using _max_size and object type
                    self._buffer.append(np.zeros((self._max_size), dtype=type(e)))
        # Append experience
        for i, e in enumerate(experience):
            self._buffer[i][self._count % self._max_size] = e
        # Increment _count
        self._count += 1        

    def sample(self, size: int) -> List:
        # Generata random indices without replacement
        indices = np.random.choice(len(self), size)
        # Take elements from each buffer using indices
        return [np.take(b, indices, axis=0) for b in self._buffer]
