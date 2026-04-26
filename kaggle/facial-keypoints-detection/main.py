import csv
import random
from collections import deque
from math import sqrt
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch._C import device as device_t
from torch.utils.data import DataLoader, Dataset


class Net(nn.Module):
    def __init__(self, output_dim: int, device: device_t):
        super().__init__()
        self.output_dim = output_dim
        self.device = device
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 10 * 10, 512)
        self.fc2 = nn.Linear(512, self.output_dim)

        self.model = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool2d(2),
            self.conv2,
            nn.ReLU(),
            nn.MaxPool2d(2),
            self.conv3,
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            self.fc1,
            nn.ReLU(),
            self.fc2,
        ).to(self.device)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.model(x)


class FacialKeypointsDataset(Dataset):
    def __init__(
        self,
        data: list[dict[str, Any]],
        fallback_value: float = -1.0,
        features: list[str] | None = None,
    ):
        self.data = data
        self.fallback_value = fallback_value
        self.features = features

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        item = self.data[idx]
        if self.features is None:
            feature_list = [k for k in item if k not in ("Image", "image_tensor")]

        else:
            feature_list = self.features
        values = []
        for feature in feature_list:
            try:
                value = float(item[feature]) / 96.0
            except ValueError:
                value = self.fallback_value
            values.append(value)
        return item["image_tensor"], torch.FloatTensor(values)


def read_data(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with open(path, "r") as f:
        dict_reader = csv.DictReader(f)
        for row in dict_reader:
            row["image_tensor"] = torch.FloatTensor(
                list(map(int, row["Image"].split(" ")))
            )
            row["image_tensor"] = row["image_tensor"].reshape(1, 96, 96)
            row["image_tensor"] /= 255.0
            result.append(row)
    return result


def get_device() -> device_t:
    device_str = "cpu"
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        device_str = "mps"
    device = device_t(device_str)
    return device


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    data_dir = Path(__file__).parent / "data"
    assert data_dir.exists(), (
        f"Data directory {data_dir} does not exist.\n"
        + "Data available at 'https://www.kaggle.com/competitions/facial-keypoints-detection/data'"
    )

    net = Net(30, device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    data_raw = read_data(data_dir / "training.csv")
    random.shuffle(data_raw)
    train_data = FacialKeypointsDataset(data_raw[:-200])
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    eval_data = FacialKeypointsDataset(data_raw[-200:])
    eval_loader = DataLoader(eval_data, batch_size=32, shuffle=False)

    loss_q = deque(maxlen=100)
    loss_fct = nn.MSELoss(reduction="none")

    for e_idx, epoch in enumerate(range(100)):
        for b_idx, (images, keypoints) in enumerate(train_loader):
            images, keypoints = images.to(device), keypoints.to(device)
            optimizer.zero_grad()
            output = net(images)
            mask = keypoints != -1.0
            loss = loss_fct(output, keypoints)
            loss = (loss * mask).sum() / mask.sum()
            loss.backward()
            loss_q.append(96.0 * sqrt(loss.item()))
            if b_idx % 50 == 0 and len(loss_q) == 100:
                print(
                    f"Epoch {e_idx}, Batch {b_idx}, Loss: {sum(loss_q) / len(loss_q):.4f}"
                )
            optimizer.step()

        with torch.no_grad():
            print("Evaluating..." + "#" * 70)
            eval_loss_q = []
            for images, keypoints in eval_loader:
                images, keypoints = images.to(device), keypoints.to(device)
                output = net(images)
                mask = keypoints != -1.0
                loss = loss_fct(output, keypoints)
                loss = (loss * mask).sum() / mask.sum()
                eval_loss_q.append(96.0 * sqrt(loss.item()))
            print(
                f"Epoch {e_idx}, Eval Loss: {sum(eval_loss_q) / len(eval_loss_q):.4f}"
            )
