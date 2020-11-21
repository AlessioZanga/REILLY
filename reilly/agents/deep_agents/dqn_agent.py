from typing import Any

import numpy as np
import tensorflow as tf

from .deep_agent import DeepAgent
from .replay_memory import ReplayMemory


class DQNAgent(DeepAgent, object):

    __slots__ = ["_batch_size", "_sample_size", "_replay_memory"]

    def __init__(
        self,
        states: Any,
        actions: int,
        alpha: float = 0.00025,
        epsilon: float = 0.99,
        gamma: float = 0.99,
        epsilon_decay: float = 1,
        batch_size: int = 32,
        sample_size: int = int(1e3),
        replay_memory_max_size: int = int(1e5),
        *args,
        **kwargs
    ):
        super().__init__(states, actions, alpha, epsilon, gamma, epsilon_decay)
        self._batch_size = batch_size
        self._sample_size = sample_size
        self._replay_memory = ReplayMemory(replay_memory_max_size)

    def _build_model(self, states: Any, actions: int) -> tf.keras.Model:
        # Build mockup state and apply tranformation
        states = self._phi(np.zeros(states)).shape

        input_frames = tf.keras.layers.Input(states)
        input_action = tf.keras.layers.Input((actions, ))
        conv_0 = tf.keras.layers.Conv2D(16, 8, strides=(4, 4), activation='relu')(input_frames)
        conv_1 = tf.keras.layers.Conv2D(32, 4, strides=(2, 2), activation='relu')(conv_0)
        flaten = tf.keras.layers.Flatten()(conv_1)
        hidden = tf.keras.layers.Dense(256, activation='relu')(flaten)
        output = tf.keras.layers.Dense(actions)(hidden)
        output_action = tf.keras.layers.Multiply()([output, input_action])
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=self._alpha, rho=0.95, epsilon=0.01)
        model = tf.keras.Model(inputs=[input_frames, input_action], outputs=output_action)
        model.compile(optimizer, loss='mse')
        return model
    
    def _phi(self, state: Any) -> Any:
        state = np.mean(state, axis=2) / 255.0      # To grayscale
        state = state[::2, ::2].astype('float32')   # Downscaling
        return state[:, :, np.newaxis]
    
    def update(self, n_S: Any, R: float, done: bool, *args, **kwargs):
        # Apply state mapping
        n_S = self._phi(n_S)

        if kwargs['training']:
            # NOTE: Apply reward clipping
            R = np.sign(R)
            # Insert experience into replay buffer
            self._replay_memory.insert((self._S, self._A, R, n_S, done))
            # Check if replay buffer contains enough experience and perform
            # graident descent every four step
            if len(self._replay_memory) >= self._sample_size and kwargs['t'] % 4 == 0:
                # Sample from replay buffer
                Ss, As, Rs, n_Ss, dones = self._replay_memory.sample(self._sample_size)
                # Predict Q values
                Qs = self._model.predict([n_Ss, np.ones_like(As)])
                # Set Q values for terminal states to zero
                Qs[dones] = 0
                # Q_values = rewards + gamma * max(Q_values)
                Qs = Rs + self._gamma * np.max(Qs, axis=1)
                # Update the model
                self._model.fit(
                    [Ss, As],
                    As * Qs[:, None],
                    epochs=1,
                    batch_size=self._batch_size,
                    verbose=0
                )
        
        # Update current state
        self._S = n_S
        self._A = self._select_action(self._S)

    def reset(self, init_state, *args, **kwargs):
        self._S = self._phi(init_state)
        self._A = self._select_action(self._S)

    def _select_action(self, state: Any) -> None:
        # Predict works on batch, inputs must be numpy arrays
        action = self._model.predict([
            np.array([state]),
            np.array([np.ones(self._actions)])
        ])[0]
        # Round, cast and extract first array from batch predict
        action = np.round(action).astype(bool)
        # NOTE: One and only one action must be selected,
        #       if not so, select a random action
        if np.sum(action) != 1:
            action = np.zeros(self._actions, dtype=bool)
            action[np.random.randint(0, len(action))] = True
        return action
