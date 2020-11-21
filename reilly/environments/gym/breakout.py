import gym

from .abstract_gym import GymEnvironment


class Breakout(GymEnvironment):

    def __init__(self):
        self._env = gym.make('BreakoutDeterministic-v4')
        self.reset()

    def run_step(self, action, *args, **kwargs):
        next_state, reward, done, _ = self._env.step(action)
        return next_state, reward, done, _
