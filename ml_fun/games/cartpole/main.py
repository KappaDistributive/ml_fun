from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def act(self, obs: np.ndarray) -> Tuple[int, torch.Tensor]:
        logits = self.forward(torch.as_tensor(obs, dtype=torch.float32))
        dist = Categorical(logits=logits)
        action = dist.sample()
        action_item = action.item()
        assert isinstance(action_item, int)
        return action_item, dist.log_prob(action)


def compute_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    returns = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.append(g)
    returns.reverse()
    returns = torch.tensor(returns, dtype=torch.float32)
    # normalize for stability
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def reinforce(
    env: gym.Env,
    policy: Policy,
    optimizer: torch.optim.Optimizer,
    num_episodes: int = 1000,
    gamma: float = 0.99,
    print_every: int = 50,
) -> List[float]:
    all_rewards: List[float] = []

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        log_probs: List[torch.Tensor] = []
        rewards: List[float] = []

        done = False
        while not done:
            action, log_prob = policy.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(float(reward))
            done = terminated or truncated

        returns = compute_returns(rewards, gamma)
        loss = -(torch.stack(log_probs) * returns).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ep_reward = sum(rewards)
        all_rewards.append(ep_reward)

        if ep % print_every == 0:
            avg = np.mean(all_rewards[-print_every:])
            print(f"Episode {ep:4d} | reward: {ep_reward:6.1f} | avg(last {print_every}): {avg:.1f}")

        if len(all_rewards) >= 100 and np.mean(all_rewards[-100:]) >= 475.0:
            print(f"Solved at episode {ep} (avg reward {np.mean(all_rewards[-100:]):.1f})")
            break

    return all_rewards


def demo_run(checkpoint_path: Path) -> None:
    render_env = gym.make("CartPole-v1", render_mode="human")
    assert render_env.observation_space.shape is not None
    obs_dim = render_env.observation_space.shape[0]
    act_dim = render_env.action_space.n # type:ignore
    policy = Policy(obs_dim, act_dim)
    policy.load_state_dict(torch.load(checkpoint_path))

    for i in range(3):
        obs, _ = render_env.reset()
        total = 0.0
        done = False
        while not done:
            action, _ = policy.act(obs)
            obs, reward, terminated, truncated, _ = render_env.step(action)
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
    plt.plot(episodes[len(episodes) - len(running_avg) :], running_avg, label="Running Average (50)", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("REINFORCE on CartPole-v1")
    plt.legend()
    plt.grid()
    plt.savefig(plot_path)
    print(f"Saved reward plot to {plot_path}")


if __name__ == "__main__":
    train = True

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = Path("ml_fun/games/cartpole/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = data_dir / f"policy_{current_time}.pt"
    plot_path = data_dir / f"rewards_{current_time}.png"

    if train:
        train_env = gym.make("CartPole-v1")
        assert train_env.observation_space.shape is not None
        obs_dim = train_env.observation_space.shape[0]
        act_dim = train_env.action_space.n # type:ignore

        policy = Policy(obs_dim, act_dim)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

        print("Training with REINFORCE...")
        rewards = reinforce(train_env, policy, optimizer)
        train_env.close()

        plot_run(plot_path, rewards)

        torch.save(policy.state_dict(), checkpoint_path)
        print(f"Saved model checkpoint to {checkpoint_path}")

    demo_run(checkpoint_path)
