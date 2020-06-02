import numpy as np

from typing import List, Dict

from .tile_coding import TileCoding


class QEstimator():
    """
    Linear action-value (q-value) function approximator for
    semi-gradient methods with state-action featurization via tile coding.
    """

    _tile_coding: TileCoding
    _num_tilings: int
    _alpha: float
    _weights: List[List]  # Every tiling have a separated list with its weights
    _max_size: int
    _have_trace: bool
    _traces: np.array

    def __init__(self, alpha: float, num_tilings: int, tiling_offset: List[float], tiles_dims: List[float], max_size: int = 4096, trace: bool = False):

        self._tile_coding = TileCoding(num_tilings, tiling_offset, tiles_dims)
        self._num_tilings = num_tilings
        # The learning rate alpha is scaled by number of tilings
        self._alpha = alpha / num_tilings
        self._weights = [np.zeros(max_size) for _ in range(num_tilings)]
        self._max_size = max_size
        self._have_trace = trace  # If trace is True initialize traces
        if self._have_trace:
            self._traces = np.zeros(max_size)

    def predict(self, state: List, action=None, number_action=None):
        """
        Predicts q-value(s) using linear FA. If action a is given then returns prediction
        for single state-action pair (s, a). Otherwise returns predictions for all actions
        in environment paired with s.
        """
        if not isinstance(state, List):
            state = [state]
        if action == None and number_action == None:
            raise "ERROR: one of action and number_action must be set"

        if action is None:
            features = [self._tile_coding.get_coordinates(state, i)
                        for i in range(number_action)]
        else:
            features = [self._tile_coding.get_coordinates(state, action)]

        return [sum([self._weights[i][feature[i]] for i in range(self._num_tilings)]) for feature in features]

    def update(self, state, action, target):
        """
        Updates the estimator parameters for a given state and action towards
        the target using the gradient update rule (and the eligibility trace if one has been set).
        """
        if not isinstance(state, List):
            state = [state]

        features = self._tile_coding.get_coordinates(state, action)
        estimation = sum([self._weights[i][features[i]]
                          for i in range(self._num_tilings)])  # Linear FA
        delta = target - estimation

        if self._have_trace:
            # self.z[features] += 1  # Accumulating trace
            self.z[features] = 1  # Replacing trace
            self.weights += self.alpha * delta * self.z
        else:
            for i in range(self._num_tilings):
                self._weights[i][features[i]] += self._alpha * delta

    def reset(self, traces_only=False):
        """
        Resets the eligibility trace (must be done at the start of every epoch) and optionally the
        weight vector (if we want to restart training from scratch).
        """
        if traces_only:
            assert self._have_trace, 'q-value estimator has no traces to reset.'
            self._traces = np.zeros(self._max_size)
        else:
            if self._have_trace:
                self._traces = np.zeros(self._max_size)
            self._weights = []