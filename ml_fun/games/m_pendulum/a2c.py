import time
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from jax import Array

# (W, b) per layer
Params = list[tuple[Array, Array]]
# (mlp_params, log_std)
Actor = tuple[Params, Array]


def model_init(
    key: Array, input_dim: int, hidden_dims: list[int], output_dim: int
) -> Params:
    """Initialize a simple MLP: list of (W, b) tuples."""
    params = []
    dims = [input_dim] + hidden_dims + [output_dim]
    for i in range(len(dims) - 1):
        key, subkey = jax.random.split(key)
        scale = jnp.sqrt(2.0 / dims[i])
        if i == len(dims) - 2:
            scale *= 0.01
        params.append(
            (
                jax.random.normal(subkey, (dims[i], dims[i + 1])) * scale,
                jnp.zeros(dims[i + 1]),
            )
        )
    return params


@jax.jit
def mlp_forward(params: Params, x: Array) -> Array:
    for i in range(len(params)):
        w, b = params[i]
        x = jnp.dot(x, w) + b
        if i < len(params) - 1:
            x = jax.nn.relu(x)
    return x


@jax.jit
def gaussian_log_prob(action: Array, mean: Array, log_std: Array) -> Array:
    std = jnp.exp(log_std)
    return jnp.sum(
        -0.5 * ((action - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)
    )


@jax.jit
def gaussian_entropy(log_std: Array) -> Array:
    return jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))


def compute_returns(rewards: list[float], gamma: float = 0.99) -> Array:
    returns = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.append(g)
    returns.reverse()
    return jnp.array(returns)


@jax.jit
def select_action(
    actor: Actor, obs: Array, sample_key: Array, action_scale: float = 3.0
) -> Array:
    raw_mean = mlp_forward(actor[0], obs)
    mean = jnp.tanh(raw_mean) * action_scale
    std = jnp.exp(actor[1])
    action = jax.random.normal(sample_key, mean.shape) * std + mean
    return action


@jax.jit
def select_action_deterministic(
    actor: Actor, obs: Array, action_scale: float = 3.0
) -> Array:
    raw_mean = mlp_forward(actor[0], obs)
    mean = jnp.tanh(raw_mean) * action_scale
    return mean


def rollout(
    env: gym.Env, actor: Actor, rng_key: Array, action_scale: float = 3.0
) -> tuple[list[Array], list[Array], list[float], Array]:
    obs, _ = env.reset()
    done = False
    observations, actions, rewards = [], [], []
    while not done:
        rng_key, sample_key = jax.random.split(rng_key)
        obs_jnp = jnp.array(obs)
        action = select_action(actor, obs_jnp, sample_key, action_scale)
        obs, reward, terminated, truncated, _ = env.step(jnp.array(action))
        done = terminated or truncated
        observations.append(obs_jnp)
        actions.append(action)
        rewards.append(float(reward))
    return observations, actions, rewards, rng_key


def make_update_fn(
    actor_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
):
    @jax.jit
    def update(
        actor,
        critic,
        actor_opt_state,
        critic_opt_state,
        obs_batch,
        act_batch,
        ret_batch,
        entropy_coeff=0.01,
        action_scale=3.0,
    ):
        # Critic loss: MSE between V(s) and returns
        def critic_loss_fn(critic_params):
            values = jax.vmap(lambda o: mlp_forward(critic_params, o))(
                obs_batch
            ).squeeze(-1)
            return jnp.mean((values - ret_batch) ** 2)

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(critic)
        critic_updates, critic_opt_state = critic_optimizer.update(
            critic_grads, critic_opt_state, critic
        )
        critic = optax.apply_updates(critic, critic_updates)

        # Compute advantages using updated critic
        values = jax.vmap(lambda o: mlp_forward(critic, o))(obs_batch).squeeze(-1)
        advantages = ret_batch - values

        # Actor loss: -E[log_pi * advantage] - entropy_coeff * entropy
        def actor_loss_fn(actor):
            actor_params, log_std = actor
            log_std = jnp.clip(log_std, -2.0, 2.0)

            def per_step_loss(obs_t, act_t, adv_t):
                raw_mean = mlp_forward(actor_params, obs_t)
                mean = jnp.tanh(raw_mean) * action_scale
                log_prob = gaussian_log_prob(act_t, mean, log_std)
                return -log_prob * jax.lax.stop_gradient(adv_t)

            policy_loss = jnp.mean(
                jax.vmap(per_step_loss)(obs_batch, act_batch, advantages)
            )
            entropy = gaussian_entropy(log_std)
            return policy_loss - entropy_coeff * entropy

        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(actor)
        actor_updates, actor_opt_state = actor_optimizer.update(
            actor_grads, actor_opt_state, actor
        )
        actor = optax.apply_updates(actor, actor_updates)

        return (
            actor,
            critic,
            actor_opt_state,
            critic_opt_state,
            actor_loss,
            critic_loss,
        )

    return update


def train_step(
    env: gym.Env,
    actor: Actor,
    critic: Params,
    update_fn,
    actor_opt_state: optax.OptState,
    critic_opt_state: optax.OptState,
    rng_key: Array,
    num_episodes: int = 16,
    gamma: float = 0.99,
    entropy_coeff: float = 0.01,
    action_scale: float = 3.0,
):
    # Data collection (Python loop — can't JIT due to env.step)
    all_obs, all_act, all_ret = [], [], []
    total_reward = 0.0
    for _ in range(num_episodes):
        observations, actions, rewards, rng_key = rollout(
            env, actor, rng_key, action_scale
        )
        returns = compute_returns(rewards, gamma)
        all_obs.extend(observations)
        all_act.extend(actions)
        all_ret.extend(returns)
        total_reward += sum(rewards)

    avg_reward = total_reward / num_episodes
    obs_batch = jnp.stack(all_obs)
    act_batch = jnp.stack(all_act)
    ret_batch = jnp.stack(all_ret)

    # Parameter update (single JIT-compiled call)
    actor, critic, actor_opt_state, critic_opt_state, actor_loss, critic_loss = (
        update_fn(
            actor,
            critic,
            actor_opt_state,
            critic_opt_state,
            obs_batch,
            act_batch,
            ret_batch,
            entropy_coeff,
            action_scale,
        )
    )

    return (
        actor,
        critic,
        actor_opt_state,
        critic_opt_state,
        rng_key,
        avg_reward,
        actor_loss,
        critic_loss,
    )


def visualize(actor: Actor, num_episodes: int = 5):
    visual_env = gym.make(
        "InvertedPendulum-v5", render_mode="human", max_episode_steps=2_000
    )
    for _ in range(num_episodes):
        obs, _ = visual_env.reset()
        done = False
        while not done:
            visual_env.render()
            time.sleep(1.0 / 30.0)
            obs_jnp = jnp.array(obs)
            action = select_action_deterministic(actor, obs_jnp)
            obs, _, terminated, truncated, _ = visual_env.step(jnp.array(action))
            done = terminated or truncated
    visual_env.close()


if __name__ == "__main__":
    env = gym.make("InvertedPendulum-v5")
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True, parents=True)
    timestamp = time.strftime("%Y%m%d%H%M%S")
    assert env.observation_space.shape is not None
    obs_dim = env.observation_space.shape[0]

    key = jax.random.PRNGKey(42)
    key, actor_key, critic_key = jax.random.split(key, 3)

    actor_params = model_init(
        actor_key, input_dim=obs_dim, hidden_dims=[64, 64], output_dim=1
    )
    log_std = jnp.zeros(1)
    actor = (actor_params, log_std)

    critic = model_init(
        critic_key, input_dim=obs_dim, hidden_dims=[64, 64], output_dim=1
    )

    num_steps = 1000
    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(optax.linear_schedule(3e-4, 1e-5, num_steps)),
    )
    critic_optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(1e-3),
    )
    actor_opt_state = actor_optimizer.init(actor)
    critic_opt_state = critic_optimizer.init(critic)
    update_fn = make_update_fn(actor_optimizer, critic_optimizer)

    rng_key = key
    for step in range(num_steps):
        (
            actor,
            critic,
            actor_opt_state,
            critic_opt_state,
            rng_key,
            avg_reward,
            actor_loss,
            critic_loss,
        ) = train_step(
            env,
            actor,
            critic,
            update_fn,
            actor_opt_state,
            critic_opt_state,
            rng_key,
        )
        if (step + 1) % 10 == 0:
            std = float(jnp.exp(actor[1][0]))
            print(
                f"Step {step + 1:4d} | avg reward: {avg_reward:6.1f} | "
                f"actor loss: {actor_loss:8.3f} | critic loss: {critic_loss:8.3f} | "
                f"std: {std:.4f}"
            )
        if (step + 1) % 50 == 0:
            checkpointer = ocp.StandardCheckpointer()

            critic_path = data_dir / f"critic_{timestamp}_step_{step + 1:04d}.npz"
            checkpointer.save(critic_path, critic)

            critic_opt_path = (
                data_dir / f"critic_opt_{timestamp}_step_{step + 1:04d}.npz"
            )
            checkpointer.save(critic_opt_path, critic_opt_state)

            actor_path = data_dir / f"actor_{timestamp}_step_{step + 1:04d}.npz"
            checkpointer.save(actor_path, actor)

            actor_opt_path = data_dir / f"actor_opt_{timestamp}_step_{step + 1:04d}.npz"
            checkpointer.save(actor_opt_path, actor_opt_state)

            # visualize(actor)
