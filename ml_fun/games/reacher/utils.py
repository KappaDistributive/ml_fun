import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def has_display() -> bool:
    """Check if a display is available for rendering."""
    try:
        import os

        return any(
            (
                os.environ.get("DISPLAY") is not None,
                os.environ.get("WAYLAND_DISPLAY") is not None,
            )
        )
    except Exception:
        return False


@dataclass
class Config:
    # Environment
    env_id: str = "Reacher-v5"
    seed: int = 42

    # Training
    total_steps: int = 300_000
    batch_size: int = 256
    learning_starts: int = 5_000  # Random actions before training starts
    train_freq: int = 1  # Train every N env steps
    gradient_steps: int = 1  # Gradient updates per train step

    # SAC hyperparameters
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Soft target update rate
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    init_alpha: float = 0.2  # Initial entropy coefficient
    target_entropy: float | None = None  # Auto-set to -action_dim

    # Network
    hidden_dim: int = 256

    # Replay buffer
    buffer_size: int = 1_000_000

    # Logging
    log_interval: int = 5_000
    eval_interval: int = 20_000
    eval_episodes: int = 10
    visual: bool = has_display()  # Whether to render during evaluation
    data_dir: Path = Path(__file__).parent / "data"  # For saving models, logs, etc.


class ReplayBuffer:
    def __init__(
        self, capacity: int, obs_dim: int, action_dim: int, device: torch.device
    ):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[idx]).to(self.device),
            torch.FloatTensor(self.actions[idx]).to(self.device),
            torch.FloatTensor(self.rewards[idx]).to(self.device),
            torch.FloatTensor(self.next_obs[idx]).to(self.device),
            torch.FloatTensor(self.dones[idx]).to(self.device),
        )


class Actor(nn.Module):
    """Gaussian policy with state-dependent mean and log_std."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        action_scale: torch.Tensor,
        action_bias: torch.Tensor,
    ):
        super().__init__()
        self.log_std_min = -5
        self.log_std_max = 2
        self.net = mlp(obs_dim, hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterisation trick
        y_t = torch.tanh(x_t)
        assert isinstance(self.action_scale, torch.Tensor) and isinstance(
            self.action_bias, torch.Tensor
        )
        action = y_t * self.action_scale + self.action_bias

        # Enforcing action bounds: log_prob correction for tanh squashing
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    """Twin Q-networks (clipped double Q to reduce overestimation)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.q1 = mlp(obs_dim + action_dim, 1, hidden_dim)
        self.q2 = mlp(obs_dim + action_dim, 1, hidden_dim)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([obs, action], dim=1)
        return self.q1(sa), self.q2(sa)


class SAC:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_space,
        cfg: Config,
        device: torch.device,
    ):
        self.cfg = cfg
        self.device = device

        # Action rescaling tensors
        action_scale = torch.FloatTensor(
            (action_space.high - action_space.low) / 2.0
        ).to(device)
        action_bias = torch.FloatTensor(
            (action_space.high + action_space.low) / 2.0
        ).to(device)

        # Networks
        self.actor = Actor(
            obs_dim, action_dim, cfg.hidden_dim, action_scale, action_bias
        ).to(device)
        self.critic = Critic(obs_dim, action_dim, cfg.hidden_dim).to(device)
        self.critic_target = Critic(obs_dim, action_dim, cfg.hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimisers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

        # Automatic entropy tuning
        self.target_entropy = cfg.target_entropy or -action_dim
        self.log_alpha = torch.tensor(
            np.log(cfg.init_alpha), requires_grad=True, device=device
        )
        self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg.lr_alpha)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action, _ = self.actor.sample(obs_t)
        return action.cpu().numpy()[0]

    def update(self, buffer: ReplayBuffer) -> dict:
        obs, actions, rewards, next_obs, dones = buffer.sample(self.cfg.batch_size)

        # ── Critic update ──────────────────────────────────────────────
        with torch.no_grad():
            next_actions, next_log_pi = self.actor.sample(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_actions)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            target_q = rewards + (1 - dones) * self.cfg.gamma * min_q_next

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── Actor update ───────────────────────────────────────────────
        new_actions, log_pi = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_actions)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_pi - min_q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ── Alpha (entropy coefficient) update ────────────────────────
        alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ── Soft target update ────────────────────────────────────────
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data
            )

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "alpha_loss": alpha_loss.item(),
        }


def mlp(
    input_dim: int, output_dim: int, hidden_dim: int, hidden_layers: int = 2
) -> nn.Sequential:
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(hidden_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def evaluate(
    agent: SAC, env_id: str, n_episodes: int = 10, seed: int = 0, visual: bool = False
) -> float:
    env = gym.make(env_id, render_mode="human" if visual else None)
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        if visual:
            env.render()
            time.sleep(0.05)
        done, ep_return = False, 0.0
        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            assert isinstance(reward, (int, float)), "Reward must be a scalar"
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)
    env.close()
    return float(np.mean(returns))


def save_model(agent: SAC, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": agent.actor.state_dict(),
            "critic": agent.critic.state_dict(),
            "critic_target": agent.critic_target.state_dict(),
            "log_alpha": agent.log_alpha,
        },
        path,
    )
    print(f"Model saved to {path}")
