import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("Using GPU:", torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Using Apple Metal Performance Shaders (MPS) backend")


class VelocityField(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.net(torch.cat([x, t], dim=-1))


def sample_checkerboard(
    n: int, width: int = 8, height: int = 8, seed: int | None = None
) -> np.ndarray:
    rng = np.random.default_rng(
        seed if seed is not None else np.random.randint(0, 2**32)
    )
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    white = np.argwhere((cols + rows) % 2 == 1)
    squares = white[rng.integers(0, len(white), n)][:, ::-1]  # flip to (x, y)
    xy = squares + rng.random((n, 2))
    xy = xy / np.array([[width, height]]) * 2 - 1
    return xy


def sample_noise(n: int, scale: float = 1.0, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(
        seed if seed is not None else np.random.randint(0, 2**32)
    )
    return rng.normal(scale=scale, size=(n, 2))


def train(
    steps: int = 5000, batch: int = 256, sigma: float = 0.0, lr: float = 1e-3
) -> nn.Module:
    model = VelocityField().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        x0, x1 = sample_noise(batch, scale=0.5), sample_checkerboard(batch, 3, 2)
        x0, x1 = torch.FloatTensor(x0).to(DEVICE), torch.FloatTensor(x1).to(DEVICE)

        t = torch.rand(batch, 1, device=DEVICE)
        xt = (1 - t) * x0 + t * x1 + sigma * torch.randn_like(x0)
        ut = x1 - x0

        loss = (model(xt, t) - ut).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1_000 == 0 or step == steps - 1:
            print(f"[CFM]] step {step:7_}  loss {loss.item():.4f}")
    return model


@torch.no_grad()
def integrate(model: nn.Module, x0: Tensor, n_steps: int) -> Tensor:
    """Euler integration of dx/dt = v(x,t) from t=0 to 1.
    Return shape: (n_steps+1, N, 2)."""
    x = x0.to(DEVICE)
    dt = 1.0 / n_steps
    traj = [x0.clone()]
    for i in range(n_steps):
        t = torch.full((x.shape[0], 1), i * dt, device=DEVICE)
        x = x + dt * model(x, t)
        traj.append(x.cpu())
    return torch.stack(traj)


if __name__ == "__main__":
    model = train(50_000, batch=1024, sigma=0.01, lr=1e-3)

    x0 = sample_noise(512, scale=0.5)
    x1 = integrate(model, torch.FloatTensor(x0), n_steps=100)
    x2 = sample_checkerboard(512, 3, 2)

    plt.figure(figsize=(6, 6))
    plt.scatter(x0[:, 0], x0[:, 1], s=2, color="gray", alpha=0.5)
    plt.plot(x1[:, :, 0], x1[:, :, 1], color="gray", lw=0.3, alpha=0.15)
    plt.scatter(x1[-1, :, 0], x1[-1, :, 1], s=4, zorder=2, color="orange")
    plt.show()
