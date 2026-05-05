import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ml_fun.lang_detect.model import ByteHybrid

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
                continue
            if label not in IDX2LANG:
                label = "unknown"
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


def evaluate(net: nn.Module, data_loader: DataLoader) -> float:
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
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print(f"Starting training at {timestamp}")
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    df = load_data_cached("train")
    net = ByteHybrid(
        num_classes=len(IDX2LANG),
        d_model=256,
        n_conv=3,
        n_attn=1,
        n_heads=4,
        conv_kernel=15,
        ngram_buckets=4096,
        ngram_dim=64,
    ).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    train_loader = DataLoader(Data(df), batch_size=128, shuffle=True)
    eval_loader = DataLoader(Data(load_data_cached("validation")), batch_size=128)

    for name, param in net.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}, {param.numel():_}")
    print(
        f"{sum(p.numel() for p in net.parameters() if p.requires_grad):_} total parameters"
    )

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
                net.eval()
                print(f"Accuracy: {100.* evaluate(net, eval_loader):.4f}%")
                net.train()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        net.eval()
        print(f"Accuracy: {100.* evaluate(net, eval_loader):.4f}%")
        net.train()
        (DATA_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            DATA_DIR / "checkpoints" / f"byte_hybrid_epoch_{timestamp}_{epoch+1}.pt",
        )
