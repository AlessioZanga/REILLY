from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import tensorflow as tf

from ..agent import Agent

class DeepAgent(Agent, ABC, object):

    __slots__ = ["_model"]

    def __init__(
        self,
        states: Any,
        actions: int,
        alpha: float,
        epsilon: float,
        gamma: float,
        epsilon_decay: float = 1,
        *args,
        **kwargs
    ):
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma
        self._e_decay = epsilon_decay
        self._actions = actions

        # Init Tensorflow GPU backend support
        try:
            for gpu in tf.config.experimental.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
        
        self._model = self._build_model(states, actions)
    
    @abstractmethod
    def _build_model(self, states: Any, actions: int) -> tf.keras.Model:
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__
