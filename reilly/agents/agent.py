from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np

from .. import backend


class Agent():

    __slots__ = [
        '_alpha', '_gamma', '_epsilon', '_e_decay', '_n_step',
        '_S', '_A', '_episode_ended', '_actions',
    ]

    _S: Any
    _A: int

    def __new__(cls, *args, **kwargs):
        if kwargs.get('backend', None) == 'cpp':
            params = {k: v for k, v in kwargs.items() if k != 'backend'}
            instan = getattr(backend, cls.__name__)
            if instan is None:
                raise NotImplementedError()
            return instan(*args, **params)
        return super(Agent, cls).__new__(cls)

    def get_action(self):
        return self._A

    @abstractmethod
    def update(self, n_S: Any, R: float, done: bool, *args, **kwargs):
        pass

    @abstractmethod
    def reset(self, init_state: Any, *args, **kwargs):
        pass

    @abstractmethod
    def _select_action(self, data: Any) -> None:
        pass
