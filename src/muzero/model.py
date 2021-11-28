from abc import ABC, abstractmethod
from typing import List, Tuple

import tensorflow as tf


class AbstractMuZeroModel(ABC):
    def __init__(self, search_depth: int):
        self.search_depth = search_depth

    @abstractmethod
    def representation_function(self, observations: tf.Tensor) -> tf.Tensor:
        """
        The representation function h_{\theta}
        :param observations:
        :return:
        """
        raise NotImplemented

    @abstractmethod
    def dynamics_function(
        self, previous_internal_state: tf.Tensor, action: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        The dynamics function g_{\theta}
        :param previous_internal_state: s^{k-1}
        :param action:  a^{k}
        :return: [immediate_reward, internal_state] = [r^{k}, s^{k}]
        """
        raise NotImplemented

    @abstractmethod
    def prediction_function(
        self, internal_state: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        The prediction function f_{\theta}
        :param internal_state: s^{k}
        :return: [policy, value] = [p^{k}, v^{k}]
        """
        raise NotImplemented


class DenseMuZeroModel(AbstractMuZeroModel):
    def __init__(
        self,
        search_depth: int,
        observation_size: int,
        action_size: int,
        state_size: int,
        hidden_layer_sizes: List[int],
    ):
        super().__init__(search_depth)
        self.observation_size = observation_size
        self.action_size = action_size
        self.state_size = state_size
        self.hidden_layer_sizes = hidden_layer_sizes

        # representation function h_{\theta}
        x = initial_observation = tf.keras.layers.Input(observation_size)
        for layer_size in self.hidden_layer_sizes:
            x = tf.keras.layers.Dense(layer_size, activation="relu")(x)
        initial_hidden_state = tf.keras.layers.Dense(self.state_size, name="s_0")(x)
        self.representation_model = tf.keras.Model(
            initial_observation, initial_hidden_state, name="h"
        )

        # dynamics function g_{\theta}
        previous_internal_state = tf.keras.layers.Input(self.state_size)
        action = tf.keras.layers.Input(self.action_size)
        x = tf.keras.layers.Concatenate()([previous_internal_state, action])
        for layer_size in self.hidden_layer_sizes:
            x = tf.keras.layers.Dense(layer_size, activation="relu")(x)
        immediate_reward = tf.keras.layers.Dense(1, name="r_k")(x)
        internal_state = tf.keras.layers.Dense(self.state_size, name="s_k")(x)
        self.dynamics_model = tf.keras.Model(
            [previous_internal_state, action],
            [immediate_reward, internal_state],
            name="g",
        )

        # prediction function f_{\theta}
        x = internal_state = tf.keras.layers.Input(self.state_size)
        for layer_size in self.hidden_layer_sizes:
            x = tf.keras.layers.Dense(layer_size, activation="relu")(x)
        policy = tf.keras.layers.Dense(self.action_size, name="p_k")(x)
        value = tf.keras.layers.Dense(1, name="v_k")(x)
        self.prediction_model = tf.keras.Model(
            internal_state, [policy, value], name="f"
        )

    def representation_function(self, observations: tf.Tensor) -> tf.Tensor:
        return self.representation_model(observations)

    def dynamics_function(
        self, previous_internal_state: tf.Tensor, action: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.dynamics_model([previous_internal_state, action])

    def prediction_function(
        self, internal_state: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.prediction_model(internal_state)
