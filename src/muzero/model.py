from abc import ABC, abstractmethod
from typing import List, Tuple

import tensorflow as tf


class AbstractMuZeroModel(ABC):
    def __init__(self, search_depth: int):
        self.search_depth = search_depth  # aka `K`

    @abstractmethod
    def representation_function(self, observation: tf.Tensor) -> tf.Tensor:
        """
        The representation function h_{\theta}
        :param observation: o^{0}
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

    @abstractmethod
    def mu_function(
        self, observation: tf.Tensor, actions: List[tf.Tensor]
    ) -> List[tf.Tensor]:
        """
        The mu function
        :param observation: o^{0}
        :param actions: [a^{1}, a^{2}, ..., a^{K}]
        :return:
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
        learning_rate: float = 1e-3,
    ):
        super().__init__(search_depth)
        self.observation_size = observation_size
        self.action_size = action_size
        self.state_size = state_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate

        # representation function h_{\theta}: (o^{0}) |---> (s^{0})
        x = initial_observation = tf.keras.layers.Input(observation_size)
        for layer_size in self.hidden_layer_sizes:
            x = tf.keras.layers.Dense(layer_size, activation="relu")(x)
        initial_hidden_state = tf.keras.layers.Dense(self.state_size, name="s^{0}")(x)
        self.representation_model = tf.keras.Model(
            initial_observation, initial_hidden_state, name="h"
        )

        # dynamics function g_{\theta}: (s^{k-1}, a^{k}) |---> (r^{k}, s^{k})
        previous_internal_state = tf.keras.layers.Input(self.state_size)
        action = tf.keras.layers.Input(self.action_size)
        x = tf.keras.layers.Concatenate()([previous_internal_state, action])
        for layer_size in self.hidden_layer_sizes:
            x = tf.keras.layers.Dense(layer_size, activation="relu")(x)
        immediate_reward = tf.keras.layers.Dense(1, name="r^{k}")(x)
        internal_state = tf.keras.layers.Dense(self.state_size, name="s^{k}")(x)
        self.dynamics_model = tf.keras.Model(
            [previous_internal_state, action],
            [immediate_reward, internal_state],
            name="g",
        )

        # prediction function f_{\theta}: (s^{k})  |---> (p^{k}, v^{k})
        x = internal_state = tf.keras.layers.Input(self.state_size)
        for layer_size in self.hidden_layer_sizes:
            x = tf.keras.layers.Dense(layer_size, activation="relu")(x)
        policy = tf.keras.layers.Dense(self.action_size, name="p^{k}")(x)
        value = tf.keras.layers.Dense(1, name="v^{k}")(x)
        self.prediction_model = tf.keras.Model(
            internal_state, [policy, value], name="f"
        )

        # mu function \mu_{\theta}: (o^{0}, a^{1}, ..., a^{K}) |---> (p^{0}, ..., p^{K}, v^{0}, ..., v^{K}, r^{1}, ..., r^{K})
        initial_observation = tf.keras.layers.Input(self.observation_size, name="o^{0}")
        previous_initial_state = self.representation_function(initial_observation)

        actions = []  # length K
        policies = []  # length K+1
        values = []  # length K+1
        rewards = []  # length K

        policy, value = self.prediction_function(previous_initial_state)
        policies.append(policy)
        values.append(values)

        for k in range(self.search_depth):
            action = tf.keras.layer.Input(self.action_size, name=f"a_{k+1}")
            actions.append(action)
            immediate_reward, internal_state = self.dynamics_model(
                previous_initial_state, action
            )
            policy, value = self.prediction_function(internal_state)
            policies.append(policy)
            values.append(values)
            rewards.append(rewards)

            previous_initial_state = initial_hidden_state

        self.mu_model = tf.keras.model(
            [initial_observation] + actions, policies + values + rewards, name="mu"
        )
        losses: List[str] = (
            (["nn.softmax_cross_entropy_with_logits"] * len(policies))
            + (["mse"] * len(values))
            + (["mse"] * len(rewards))
        )

        self.mu_model.compile(tf.keras.optimizers.Adam(self.learning_rate), losses)

    def representation_function(self, observation: tf.Tensor) -> tf.Tensor:
        return self.representation_model(observation)

    def dynamics_function(
        self, previous_internal_state: tf.Tensor, action: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.dynamics_model([previous_internal_state, action])

    def prediction_function(
        self, internal_state: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.prediction_model(internal_state)

    def mu_function(
        self, observation: tf.Tensor, actions: List[tf.Tensor]
    ) -> List[tf.Tensor]:
        return self.mu_model([observation] + actions)
