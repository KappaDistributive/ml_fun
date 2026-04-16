import gymnasium as gym
import jax
import jax.numpy as jnp
import optax


def model_init(
    input_dim: int, hidden_dims: list[int], output_dim: int
) -> list[tuple[tuple[str, jnp.ndarray], tuple[str, jnp.ndarray]]]:
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, num=len(hidden_dims) + 2)
    params = []
    if not hidden_dims:
        params.append(
            (
                ("W1", jax.random.normal(keys[0], (input_dim, output_dim))),
                ("b1", jnp.zeros(output_dim)),
            )
        )
        return params
    for index in range(len(hidden_dims)):
        if index == 0:
            params.append(
                (
                    ("W1", jax.random.normal(keys[0], (input_dim, hidden_dims[0]))),
                    ("b1", jnp.zeros(hidden_dims[0])),
                )
            )
        elif index < len(hidden_dims):
            params.append(
                (
                    (
                        f"W{index+1}",
                        jax.random.normal(
                            keys[index + 1],
                            (hidden_dims[index - 1], hidden_dims[index]),
                        ),
                    ),
                    (f"b{index+1}", jnp.zeros(hidden_dims[index])),
                )
            )
    params.append(
        (
            (
                f"W{len(hidden_dims)+1}",
                jax.random.normal(
                    keys[len(hidden_dims) + 1],
                    (hidden_dims[len(hidden_dims) - 1], output_dim),
                ),
            ),
            (f"b{len(hidden_dims)+1}", jnp.zeros(output_dim)),
        )
    )
    return params


def model_forward(
    params: list[tuple[tuple[str, jnp.ndarray], tuple[str, jnp.ndarray]]],
    x: jnp.ndarray,
) -> jnp.ndarray:
    for index in range(1, len(params)):
        w = params[index][0][1]
        b = params[index][1][1]
        x = jnp.dot(x, w) + b
        if index < len(params) - 1:
            x = jax.nn.relu(x)
    return x


def train_step() -> tuple[optax.Params, optax.Params, optax.OptState]: ...


def model_description(
    model: tuple[
        list[tuple[tuple[str, jnp.ndarray], tuple[str, jnp.ndarray]]],
        tuple[str, jnp.ndarray],
    ],
) -> None:
    params, std_param = model
    print("Model description:")
    for index in range(len(params)):
        print(
            f"{params[index][0][0]}: {params[index][0][1].shape}\t{params[index][1][0]}: {params[index][1][1].shape}"
        )
    print(f"{std_param[0]}: {std_param[1].shape}")


if __name__ == "__main__":
    env = gym.make("InvertedPendulum-v5")
    obs, _ = env.reset(seed=0)

    assert env.observation_space.shape is not None, "Observation space must be a vector"
    obs_dim = env.observation_space.shape[0]
    assert isinstance(obs_dim, int), "Observation space must be a vector"
    assert env.action_space.shape is not None, "Action space must be a vector"
    action_dim = env.action_space.shape[0]
    assert isinstance(action_dim, int), "Action space must be a vector"

    params = model_init(input_dim=obs_dim, hidden_dims=[64, 64], output_dim=1)
    std_param = ("std", jnp.ones(1))
    model = (params, std_param)
    model_description(model)
