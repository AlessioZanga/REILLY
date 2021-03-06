from typing import Tuple

import gym

import numpy as np

from .abstract_gym import GymEnvironment


class AtariEnvironment(GymEnvironment):

    def __init__(self, name: str, history: int = 4):
        self._name = name
        self._history = history
        self._env = gym.make(name)
        self.reset()
    
    @property
    def states(self) -> Tuple[int]:
        return (*self._env.observation_space.shape, self._history)

    def run_step(self, action, *args, **kwargs):
        action = action.astype(bool)
        n_state, n_reward, n_done, other = np.empty(self.states), 0, False, None
        for i in range(self._history):
            state, reward, done, other = self._env.step(action)
            n_state[..., i] = state
            n_reward += reward
            n_done |= done
        return n_state, n_reward, n_done, other
    
    def reset(self, *args, **kwargs):
        reset = self._env.reset()
        reset = np.stack([reset] * self._history, axis=-1)
        return reset
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}-{self._name}"
