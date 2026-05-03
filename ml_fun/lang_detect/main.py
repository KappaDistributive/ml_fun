from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA_DIR: Path = Path(__file__).parent / "data"
DATA_BASE_URL: str = "hf://datasets/PleIAs/CommonLingua-Train/"
DATA_SPLITS: dict[str, str] = {"train": "train.parquet", "validation": "val.parquet"}
LANG_CODES_PATH = Path(__file__).parent / "lang_codes.txt"

IDX2LANG: list[str] = []
with open(LANG_CODES_PATH, "r") as f:
    for line in f:
        lang = line.strip()
        assert lang, "Empty line found in lang_codes.txt -- aborting!"
        IDX2LANG.append(lang)
    assert "unknown" not in IDX2LANG, "Found 'nan' in lang_codes.txt -- aborting!"
    IDX2LANG.append("unknown")
LANG2IDX: dict[str, int] = {lang: idx for idx, lang in enumerate(IDX2LANG)}


class Embed(nn.Module):
    def __init__(self, num_buckets: int, embedding_dim: int, n: int = 3):
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_buckets, embedding_dim)
        self.n = n

    def forward(self, byte_ids: torch.LongTensor) -> torch.FloatTensor:
        # shape of byte_ids: (batch_size =: b, max_len =: t)
        b, t = byte_ids.shape
        clamped = byte_ids.clamp(max=255)
        # padded extends clamped by enough zeros to allow for n-gram addition
        # shape of padded: (batch_size, max_len + n - 1)
        padded = F.pad(clamped, (0, self.n - 1), value=0)
        # h[0,0] = padded[0,0] * 256**(n-1) + padded[0,1] * 256**(n-2) + .. + padded[0,n-1]
        h = torch.zeros(b, t, dtype=torch.long, device=byte_ids.device)
        for i in range(self.n):
            h = h * 256 + padded[:, i : i + t]
        # output shape: (batch_size, max_len, embedding_dim)
        return self.embedding(h % self.num_buckets)


class Net(nn.Module):
    def __init__(
        self, num_buckets: int, max_length: int, embedding_dim: int, num_classes: int
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.embed = Embed(self.num_buckets, self.embedding_dim)
        self.fc = nn.Linear(self.max_length * self.embedding_dim, self.num_classes)

    def forward(self, byte_ids: torch.LongTensor) -> torch.FloatTensor:
        x = self.embed(byte_ids)  # (batch_size, max_len, embedding_dim)
        x = x.view(x.size(0), -1)  # flatten to (batch_size, max_len * embedding_dim)
        x = self.fc(x)  # (batch_size, num_classes)
        return x


def embed(
    texts: list[str], max_len: int = 512, verbose: bool = False
) -> torch.LongTensor:
    out = np.full((len(texts), max_len), fill_value=256, dtype=np.int64)
    for text_idx, text in enumerate(texts):
        if verbose and len(text) > max_len:
            print(
                f"Warning: Text {text[:10]} exceeds max length of {max_len} bytes and will be truncated."
            )
        raw = text.encode("utf-8", errors="replace")[:max_len]
        out[text_idx, : len(raw)] = list(raw)
    result = torch.from_numpy(out.astype(np.int64))
    assert isinstance(
        result, torch.LongTensor
    ), f"Expected result to be a LongTensor, but got {type(result)}"
    return result


def load_data_cached(split: str) -> pl.DataFrame:
    """Load the specified data split, caching it locally if not already cached."""
    if split not in DATA_SPLITS:
        raise ValueError(
            f"Invalid split '{split}'. Valid splits are: {list(DATA_SPLITS.keys())}"
        )

    local_path = DATA_DIR / DATA_SPLITS[split]
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {split} split from Hugging Face...")
        df = pl.read_parquet(DATA_BASE_URL + DATA_SPLITS[split])
        df.write_parquet(local_path)
        print(f"Saved {split} split to cache at {local_path}")
    else:
        print(f"Loading {split} split from cache...")
        df = pl.read_parquet(local_path)

    return df


class Data(Dataset):
    def __init__(self, df: pl.DataFrame):
        texts = df["text"].to_list()
        labels = df["lang"].to_list()

        self.texts: list[str] = []
        self.labels: list[str] = []

        for idx in range(len(texts)):
            text = texts[idx]
            if not text or not isinstance(text, str):
                continue
            label = labels[idx]
            if not label or not isinstance(label, str):
                if label not in IDX2LANG:
                    label = "unknown"
                continue
            self.texts.append(text)
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        text = self.texts[idx]
        label = self.labels[idx]
        byte_ids = embed([text])[0]  # shape: (max_len,)
        label_idx = LANG2IDX[label]
        return byte_ids, label_idx


def evaluate(net: Net, data_loader: DataLoader) -> float:
    correct = 0
    total = 0
    device = next(net.parameters()).device
    with torch.no_grad():
        for byte_ids, labels in data_loader:
            logits = net(byte_ids.to(device))
            predictions = torch.argmax(logits, dim=1).cpu()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


if __name__ == "__main__":
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    df = load_data_cached("train")
    net = Net(4096, 512, 64, len(IDX2LANG)).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    train_loader = DataLoader(Data(df), batch_size=128, shuffle=True)
    eval_loader = DataLoader(Data(load_data_cached("validation")), batch_size=128)

    for epoch in range(10):
        net.train()
        total_loss = 0.0
        p_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        num_steps = len(train_loader)
        for b_idx, (byte_ids, labels) in enumerate(p_bar):
            optimizer.zero_grad()
            logits = net(byte_ids.to(device))
            loss = F.cross_entropy(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (b_idx + 1) % (num_steps // 10) == 0:
                print(f"Batch {b_idx+1} Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        net.eval()
        print(f"Accuracy: {100.* evaluate(net, eval_loader):.4f}%")
        net.train()
