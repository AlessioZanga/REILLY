from typing import Tuple

import gym

from .abstract_gym import GymEnvironment


class Breakout(GymEnvironment):

    def __init__(self):
        self._env = gym.make('BreakoutDeterministic-v4')
        self.reset()
    
    @property
    def states(self) -> Tuple[int]:
        return self._env.observation_space.shape

    def run_step(self, action, *args, **kwargs):
        next_state, reward, done, _ = self._env.step(action)
        return next_state, reward, done, _
