from abc import ABC, abstractmethod
from typing import List, Tuple

import tensorflow as tf


class AbstractMuZeroModel(ABC):
    def __init__(
        self,
        lookahead_range: int,
        observation_size: int,
        action_size: int,
        state_size: int,
    ):
        """
        :param lookahead_range: Lookahead range, i.e. the number of actions performed in each sequence.
        :param observation_size: The size of the observation space -- it is assumed to be one-dimensional for now.
        :param action_size: The total number of possible actions in each step.
        :param state_size: The size of the internal (hidden) state.
        """
        self.lookahead_range = lookahead_range  # aka `K`
        self.observation_size = observation_size
        self.action_size = action_size

    @abstractmethod
    def representation_function(self, observation: tf.Tensor) -> tf.Tensor:
        """
        The representation function `h_{theta}`.
        :param observation: o^{0} of shape (batch_size, observation_size).
        :return: s^{0} of shape (batch_size, state_size).
        """
        raise NotImplemented

    @abstractmethod
    def dynamics_function(
        self, previous_internal_state: tf.Tensor, action: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        The dynamics function `g_{theta}`.
        :param previous_internal_state: s^{k-1}
        :param action:  a^{k}
        :return: [immediate_reward, internal_state] = [r^{k}, s^{k}], where
        - r^{k} has shape (batch_size, 1) and
        - r^{k} has shape (batch_size, state_size).
        """
        raise NotImplemented

    @abstractmethod
    def prediction_function(
        self, internal_state: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        The prediction function `f_{theta}`.
        :param internal_state: s^{k}
        :return: [policy, value] = [p^{k}, v^{k}], where
        - p^{k} has shape (batch_size, action_size) and
        - v^{k} has shape (batch_size, 1).
        """
        raise NotImplemented

    @abstractmethod
    def mu_function(
        self, observation: tf.Tensor, actions: List[tf.Tensor]
    ) -> List[tf.Tensor]:
        """
        The mu function.
        :param observation: o^{0}
        :param actions: [a^{1}, a^{2}, ..., a^{K}]
        :return: [p^{0}, ..., p^{K}, v^{0}, ..., v^{K}, r^{1}, ..., r^{K}], where
        - p^{i} is the policy for the i-th step and has shape (batch_size, action_size),
        - v^{i} is the value for the i-th step and has shape (batch_size, 1),
        - r^{i} is the immediate reward for the i-th step and has shape (batch_size, 1).
        """
        raise NotImplemented

    @abstractmethod
    def train_on_batch(
        self, observation: tf.Tensor, actions: List[tf.Tensor], target: List[tf.Tensor]
    ) -> float:
        raise NotImplemented


class DenseMuZeroModel(AbstractMuZeroModel):
    def __init__(
        self,
        lookahead_range: int,
        observation_size: int,
        action_size: int,
        state_size: int,
        hidden_layer_sizes: List[int],
        learning_rate: float = 1e-3,
    ):
        """
        :param lookahead_range: Lookahead range, i.e. the number of actions performed in each sequence.
        :param observation_size: The size of the observation space -- it is assumed to be one-dimensional for now.
        :param action_size: The total number of possible actions in each step.
        :param state_size: The size of the internal (hidden) state.
        :param hidden_layer_sizes: Layer sizes of hidden layers in internal dense feed-forward networks.
        :param learning_rate: Learning rate.
        """
        # TODO: Accept an optimizer instead of a learning rate.
        super().__init__(lookahead_range, observation_size, action_size, state_size)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.losses: List[float] = []

        # representation function h_{theta}: (o^{0}) |---> (s^{0})
        initial_observation = tf.keras.Input(self.observation_size)
        x = initial_observation
        for layer_size in self.hidden_layer_sizes:
            x = tf.keras.layers.Dense(layer_size, activation="relu")(x)
        initial_hidden_state = tf.keras.layers.Dense(self.state_size, name="s_0")(x)
        self.representation_model = tf.keras.Model(
            initial_observation, initial_hidden_state, name="h"
        )

        # dynamics function g_{theta}: (s^{k-1}, a^{k}) |---> (r^{k}, s^{k})
        previous_internal_state = tf.keras.Input(self.state_size)
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

        # prediction function f_{theta}: (s^{k})  |---> (p^{k}, v^{k})
        x = internal_state = tf.keras.Input(self.state_size)
        for layer_size in self.hidden_layer_sizes:
            x = tf.keras.layers.Dense(layer_size, activation="relu")(x)
        policy = tf.keras.layers.Dense(self.action_size, name="p_k")(x)
        value = tf.keras.layers.Dense(1, name="v_k")(x)
        self.prediction_model = tf.keras.Model(
            internal_state, [policy, value], name="f"
        )

        # mu function \mu_{theta}: (o^{0}, a^{1}, ..., a^{K}) |---> (p^{0}, ..., p^{K}, v^{0}, ..., v^{K}, r^{1}, ..., r^{K})
        initial_observation = tf.keras.Input(self.observation_size, name="o_0")
        previous_internal_state = self.representation_function(initial_observation)

        actions = []  # length K
        policies = []  # length K+1
        values = []  # length K+1
        rewards = []  # length K

        policy, value = self.prediction_function(previous_internal_state)
        policies.append(policy)
        values.append(value)

        for k in range(self.lookahead_range):
            action = tf.keras.Input(self.action_size, name=f"a_{k+1}")
            actions.append(action)
            immediate_reward, internal_state = self.dynamics_model(
                [previous_internal_state, action]
            )
            policy, value = self.prediction_function(internal_state)
            policies.append(policy)
            values.append(value)
            rewards.append(immediate_reward)

            previous_internal_state = internal_state

        self.mu_model = tf.keras.Model(
            inputs=[initial_observation] + actions,
            outputs=policies + values + rewards,
            name="mu",
        )

        losses: List[str] = (
            (
                [tf.keras.losses.CategoricalCrossentropy(from_logits=True)]
                * len(policies)
            )
            + ([tf.keras.losses.MeanSquaredError()] * len(values))
            + ([tf.keras.losses.MeanSquaredError()] * len(rewards))
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

    def train_on_batch(
        self, observation: tf.Tensor, actions: List[tf.Tensor], targets: List[tf.Tensor]
    ) -> float:
        loss = self.mu_model.train_on_batch(x=[observation] + actions, y=targets)
        self.losses.append(loss)
        return loss
