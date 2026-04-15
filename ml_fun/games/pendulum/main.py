from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Actor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((1,), -0.5))

    def forward(self, obs: torch.Tensor) -> Normal:
        h = self.net(obs)
        mean = 2.0 * torch.tanh(self.mean_head(h))
        # floor std at ~0.2 to maintain exploration
        log_std = torch.clamp(self.log_std, min=-1.5)
        std = log_std.exp().expand_as(mean)
        return Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def collect_rollouts(
    env: gym.Env,
    actor: Actor,
    critic: Critic,
    num_episodes: int,
    device: torch.device,
    reward_scale: float = 0.1,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[float]
]:
    """Collect episodes and compute GAE. Returns tensors ready for PPO updates."""
    all_obs: List[torch.Tensor] = []
    all_actions: List[torch.Tensor] = []
    all_log_probs: List[torch.Tensor] = []
    all_advantages: List[torch.Tensor] = []
    all_returns: List[torch.Tensor] = []
    ep_rewards: List[float] = []

    gamma = 0.99
    lam = 0.95

    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_obs, ep_actions, ep_log_probs, ep_values, ep_rewards_raw = (
            [],
            [],
            [],
            [],
            [],
        )

        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                dist = actor(obs_t)
                value = critic(obs_t)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            ep_obs.append(obs_t)
            ep_actions.append(action.squeeze())
            ep_log_probs.append(log_prob.squeeze())
            ep_values.append(value.squeeze())

            action_np = np.array([action.clamp(-2.0, 2.0).item()])
            obs, reward, terminated, truncated, _ = env.step(action_np)
            ep_rewards_raw.append(float(reward))
            done = terminated or truncated

        ep_rewards.append(sum(ep_rewards_raw))

        # compute GAE on scaled rewards
        scaled_rewards = [r * reward_scale for r in ep_rewards_raw]
        values = torch.stack(ep_values)
        next_value = torch.tensor(0.0, device=device)
        vals = torch.cat([values, next_value.unsqueeze(0)])

        advantages = torch.zeros(len(scaled_rewards), device=device)
        gae = 0.0
        for t in reversed(range(len(scaled_rewards))):
            delta = scaled_rewards[t] + gamma * vals[t + 1] - vals[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae

        returns = advantages + values

        all_obs.append(torch.stack(ep_obs))
        all_actions.append(torch.stack(ep_actions))
        all_log_probs.append(torch.stack(ep_log_probs))
        all_advantages.append(advantages)
        all_returns.append(returns)

    return (
        torch.cat(all_obs),
        torch.cat(all_actions),
        torch.cat(all_log_probs),
        torch.cat(all_advantages),
        torch.cat(all_returns),
        ep_rewards,
    )


def ppo(
    env: gym.Env,
    actor: Actor,
    critic: Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_updates: int = 2000,
    episodes_per_update: int = 20,
    ppo_epochs: int = 10,
    minibatch_size: int = 256,
    clip_eps: float = 0.2,
    entropy_coeff: float = 1e-3,
    print_every: int = 5,
) -> List[float]:
    all_rewards: List[float] = []

    for update in range(1, num_updates + 1):
        obs, actions, old_log_probs, advantages, returns, ep_rewards = collect_rollouts(
            env, actor, critic, episodes_per_update, device
        )
        all_rewards.extend(ep_rewards)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update: multiple epochs over the same data
        n = obs.shape[0]
        for _ in range(ppo_epochs):
            indices = torch.randperm(n)
            for start in range(0, n, minibatch_size):
                idx = indices[start : start + minibatch_size]

                mb_obs = obs[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]

                # actor
                dist = actor(mb_obs)
                new_log_probs = dist.log_prob(mb_actions.unsqueeze(-1)).squeeze(-1)
                ratio = (new_log_probs - mb_old_log_probs).exp()

                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss -= entropy_coeff * dist.entropy().mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                actor_optimizer.step()

                # critic
                values = critic(mb_obs).squeeze(-1)
                critic_loss = nn.functional.mse_loss(values, mb_returns)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                critic_optimizer.step()

        if update % print_every == 0:
            avg_batch = np.mean(ep_rewards)
            avg_recent = np.mean(all_rewards[-100:])
            total_eps = update * episodes_per_update
            print(
                f"Update {update:4d} (ep {total_eps:5d}) | "
                f"batch: {avg_batch:7.1f} | "
                f"avg(100): {avg_recent:7.1f} | "
                f"std: {actor.log_std.exp().item():.3f}"
            )

        if len(all_rewards) >= 100 and np.mean(all_rewards[-100:]) >= -200.0:
            total_eps = update * episodes_per_update
            print(
                f"Solved at update {update} (ep {total_eps}) "
                f"(avg reward {np.mean(all_rewards[-100:]):.1f})"
            )
            break

    return all_rewards


def demo_run(
    env_name: str, obs_dim: int, checkpoint_path: Path, device: torch.device
) -> None:
    render_env = gym.make(env_name, render_mode="human")
    actor = Actor(obs_dim)
    actor.load_state_dict(torch.load(checkpoint_path, map_location=device))
    actor.to(device)

    for i in range(3):
        obs, _ = render_env.reset()
        total = 0.0
        done = False
        while not done:
            with torch.no_grad():
                dist = actor(torch.as_tensor(obs, dtype=torch.float32, device=device))
            action = dist.mean.clamp(-2.0, 2.0).item()
            obs, reward, terminated, truncated, _ = render_env.step(np.array([action]))
            assert isinstance(reward, (float, int))
            total += reward
            done = terminated or truncated
        print(f"Render episode {i + 1}: reward = {total}")
    render_env.close()


def plot_run(plot_path: Path, rewards: List[float]) -> None:
    episodes = np.arange(1, len(rewards) + 1)
    running_avg = np.convolve(rewards, np.ones(50) / 50, mode="valid")

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, rewards, label="Episode Reward", alpha=0.6)
    plt.plot(
        episodes[len(episodes) - len(running_avg) :],
        running_avg,
        label="Running Average (50)",
        color="red",
    )
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO on Pendulum-v1")
    plt.legend()
    plt.grid()
    plt.savefig(plot_path)
    print(f"Saved reward plot to {plot_path}")


if __name__ == "__main__":
    env_name = "Pendulum-v1"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = data_dir / f"policy_{current_time}.pt"
    plot_path = data_dir / f"rewards_{current_time}.png"

    device = get_device()
    print(f"Using device: {device}")

    train_env = gym.make(env_name)

    assert train_env.observation_space.shape is not None
    obs_dim = train_env.observation_space.shape[0]

    actor = Actor(obs_dim).to(device)
    critic = Critic(obs_dim).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    print("Training with PPO (Gaussian policy)...")
    rewards = ppo(train_env, actor, critic, actor_optimizer, critic_optimizer, device)
    train_env.close()

    plot_run(plot_path, rewards)

    torch.save(actor.state_dict(), checkpoint_path)
    print(f"Saved actor checkpoint to {checkpoint_path}")

    demo_run(env_name, obs_dim, checkpoint_path, device)
