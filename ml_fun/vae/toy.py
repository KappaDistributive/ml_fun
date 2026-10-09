import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def generate_blob(x: float, y: float, width: int, height: int) -> np.ndarray:
    assert 0.0 <= x <= 1.0, "x must be in the range [0, 1]"
    assert 0.0 <= y <= 1.0, "y must be in the range [0, 1]"
    assert width > 0, "width must be positive"
    assert height > 0, "height must be positive"
    center_x = x * width
    center_y = y * height
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    # https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
    blob = np.exp(
        -((y_grid - center_y) ** 2 + (x_grid - center_x) ** 2)
        / (2 * (0.1 * min(width, height)) ** 2)
    )
    return blob


def make_blobs(
    num_blobs: int, width: int, height: int
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    blobs = []
    coords = []
    for _ in range(num_blobs):
        x, y = np.random.rand(2)
        blob = generate_blob(x, y, width, height)
        blobs.append(blob)
        coords.append((x, y))

    return np.stack(blobs, dtype=np.float32), coords


class VAE(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(400, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.to_mu = nn.Linear(hidden, 2)
        self.to_logvar = nn.Linear(hidden, 2)
        self.decoder = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 400),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h)

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self._reparameterize(mu, logvar)
        logits = self.decoder(z)
        return logits, mu, logvar


def vae_loss(
    logits: Tensor, x: Tensor, mu: Tensor, logvar: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + kl, recon, kl


if __name__ == "__main__":
    WIDTH, HEIGHT = 20, 20
    ema_weight = 0.98

    vae = VAE()
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    # training
    batch_size = 256
    num_epochs = 1_000
    ema = None
    for epoch in range(num_epochs):
        x = torch.tensor(make_blobs(batch_size, WIDTH, HEIGHT)[0]).reshape(
            batch_size, -1
        )
        logits, mu, logvar = vae(x)
        loss, recon, kl = vae_loss(logits, x, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ema = (
            loss.item()
            if ema is None
            else ema_weight * ema + (1 - ema_weight) * loss.item()
        )
        if epoch % 100 == 0:
            print(
                f"Epoch {epoch}: Loss={ema:.4f}, "
                f"Recon={recon.item():.4f}, KL={kl.item():.4f}"
            )

    # visual inspection
    x_ref, coords_ref = make_blobs(1000, WIDTH, HEIGHT)
    x_ref = torch.tensor(x_ref).reshape(1000, -1)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=10, ncols=2, figsize=(2, 10))
    with torch.no_grad():
        logits_ref, _, _ = vae(x_ref)
        pixels_ref = torch.sigmoid(logits_ref)
    for i in range(10):
        axes[i, 0].imshow(x_ref[i].reshape(WIDTH, HEIGHT), cmap="gray")
        axes[i, 1].imshow(
            pixels_ref[i].detach().numpy().reshape(WIDTH, HEIGHT), cmap="gray"
        )
    plt.show()
