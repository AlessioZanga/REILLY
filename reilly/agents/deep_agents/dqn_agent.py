from typing import Any

import os
import numpy as np
import tensorflow as tf

from .deep_agent import DeepAgent
from .replay_memory import ReplayMemory


class DQNAgent(DeepAgent, object):

    __slots__ = ["_batch_size", "_replay_memory", "_replace_size", "_model_star", "_steps", "_fading"]

    def __init__(
        self,
        states: Any,
        actions: int,
        alpha: float = 0.00025,
        epsilon: float = 0.99,
        gamma: float = 0.99,
        epsilon_decay: float = 0.99995,
        batch_size: int = 32,
        replace_size: int = int(1e5),
        replay_memory_max_size: int = int(1e5),
        load_model: str = None,
        *args,
        **kwargs
    ):
        self._fading = np.linspace(0.4, 1, states[-1])
        self._fading = self._fading[None, None, ...]
        
        super().__init__(states, actions, alpha, epsilon, gamma, epsilon_decay)
        self._batch_size = batch_size
        self._replay_memory = ReplayMemory(replay_memory_max_size)

        self._steps = 0
        self._replace_size = replace_size
        self._model_star = self._build_model(states, actions)

        if load_model:
            self._model = tf.keras.models.load_model(load_model)
            self._model_star = tf.keras.models.load_model(load_model)

    def _build_model(self, states: Any, actions: int) -> tf.keras.Model:
        # Build mockup state and apply tranformation
        states = self._phi(np.zeros(states)).shape
        model = tf.keras.Sequential([
            tf.keras.layers.Input(states),
            tf.keras.layers.Conv2D(16, 8, strides=(4, 4), activation='relu'),
            tf.keras.layers.Conv2D(32, 4, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(actions)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=self._alpha)
        model.compile(optimizer, loss='mse')
        return model
    
    def _phi(self, state: Any) -> Any:
        state = state[::2, ::2]                     # Downscale
        state = np.sum(state, +2) * self._fading   # Grayscale and fading
        state = np.sum(state, -1)                   # History aggregation
        state = state / np.max(state)               # Normalization
        state = state.astype(np.float32)            # Quantization
        state = np.expand_dims(state, -1)           # Channel ordering
        return state
    
    def update(self, n_S: Any, R: float, done: bool, *args, **kwargs):
        # Apply state mapping
        n_S = self._phi(n_S)

        if kwargs['training']:
            # Insert experience into replay buffer
            self._replay_memory.insert((self._S, self._A, R, n_S, done))
            # Check if replay buffer contains enough experience
            if len(self._replay_memory) >= self._batch_size:
                # Sample from replay buffer
                Ss, As, Rs, n_Ss, dones = self._replay_memory.sample(self._batch_size)
                # Predict Q values
                Qs = self._model_star.predict_on_batch(n_Ss)
                # Set Q values for terminal states to zero
                Qs[dones] = 0
                # Q_values = rewards + gamma * max(Q_values)
                Qs = Rs + self._gamma * np.max(Qs, axis=1)
                # Update the model
                self._model.train_on_batch(Ss, As * Qs[:, None])

            # Replace model if reached replace size
            if not self._steps % self._replace_size:
                self._model_star.set_fading(self._model.get_fading())
            # Increment global steps count
            self._steps += 1

            # If episode done
            if done:
                if self._epsilon > 0.1:
                    self._epsilon *= self._e_decay

        # Update current state
        self._S = n_S
        self._A = self._select_action(self._S)

    def reset(self, init_state, *args, **kwargs):
        self._S = self._phi(init_state)
        self._A = self._select_action(self._S)

    def _select_action(self, state: Any) -> None:
        action = np.zeros(self._actions, dtype=np.float32)
        # With probability _epsilon select a random action
        if np.random.random() > self._epsilon:
            Q = self._model.predict(state[None, ...])
            action[np.argmax(Q)] = 1
        else:
            action[np.random.randint(1, len(action))] = 1
        return action
