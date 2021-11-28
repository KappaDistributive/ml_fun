import gym
import tensorflow as tf

from src.muzero.model import DenseMuZeroModel
from src.muzero.search import naive_search

if __name__ == "__main__":
    environment = gym.make("CartPole-v0")
    observation = environment.reset()
    # print(environment.action_space)
    # print(observation, )
    model = DenseMuZeroModel(
        5,
        environment.observation_space.shape[0],
        environment.action_space.n,
        32,
        [32, 32],
    )
    # print(model.mu_model.summary())

    print(naive_search(model, tf.convert_to_tensor(observation)))
