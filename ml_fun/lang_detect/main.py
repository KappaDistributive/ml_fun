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


class ByteNgramEmbed(nn.Module):

    def __init__(self, num_buckets=4096, embed_dim=64, n=3):
        super().__init__()
        self.n = n
        self.num_buckets = num_buckets
        self.embed = nn.Embedding(num_buckets, embed_dim)

    def forward(self, byte_ids):
        B, T = byte_ids.shape
        clamped = byte_ids.clamp(max=255)
        padded = F.pad(clamped, (0, self.n - 1), value=0)
        # h[0,0] = padded[0,0] * 256**(n-1) + padded[0,1] * 256**(n-2) + .. + padded[0,n-1]
        h = torch.zeros(B, T, dtype=torch.long, device=byte_ids.device)
        for i in range(self.n):
            h = h * 257 + padded[:, i : i + T]
        return self.embed(h % self.num_buckets)


class ByteConvBlock(nn.Module):

    def __init__(self, d_model, kernel_size=15, expand=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model)
        self.norm2 = nn.LayerNorm(d_model)
        ffn = d_model * expand
        self.ffn_gate = nn.Linear(d_model, ffn, bias=False)
        self.ffn_up = nn.Linear(d_model, ffn, bias=False)
        self.ffn_down = nn.Linear(ffn, d_model, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm1(x).transpose(1, 2)
        x = F.pad(x, (self.pad, 0))
        x = F.silu(self.conv(x)).transpose(1, 2)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn_down(F.silu(self.ffn_gate(x)) * self.ffn_up(x))
        return residual + x


def _rope(q, k):
    head_dim = q.shape[-1]
    seq_len = q.shape[-2]
    freqs = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2, device=q.device).float() / head_dim)
    )
    t = torch.arange(seq_len, device=q.device)
    a = torch.outer(t, freqs)
    cos = a.cos().to(q.dtype)
    sin = a.sin().to(q.dtype)

    def rot(x):
        x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    return rot(q), rot(k)


class ByteAttnBlock(nn.Module):

    def __init__(self, d_model, n_heads=4, expand=2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        ffn = d_model * expand
        self.ffn_gate = nn.Linear(d_model, ffn, bias=False)
        self.ffn_up = nn.Linear(d_model, ffn, bias=False)
        self.ffn_down = nn.Linear(ffn, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        residual = x
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
        q, k = _rope(q, k)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        x = residual + self.out_proj(out)

        residual = x
        h = self.norm2(x)
        h = self.ffn_down(F.silu(self.ffn_gate(h)) * self.ffn_up(h))
        return residual + h


class ByteHybrid(nn.Module):

    def __init__(
        self,
        num_classes,
        d_model=256,
        n_conv=3,
        n_attn=1,
        n_heads=4,
        ffn_expand=2,
        max_len=512,
        conv_kernel=15,
        ngram_buckets=0,
        ngram_dim=64,
    ):
        super().__init__()
        self.max_len = max_len

        # Byte values 0–255 plus index 256 = padding token
        self.embed = nn.Embedding(257, d_model, padding_idx=256)

        self.ngram_embed = None
        if ngram_buckets > 0:
            self.ngram_embed = ByteNgramEmbed(ngram_buckets, ngram_dim, n=3)
            self.ngram_proj = nn.Linear(ngram_dim, d_model, bias=False)

        self.conv_layers = nn.ModuleList(
            [ByteConvBlock(d_model, conv_kernel, ffn_expand) for _ in range(n_conv)]
        )
        self.attn_layers = nn.ModuleList(
            [ByteAttnBlock(d_model, n_heads, ffn_expand) for _ in range(n_attn)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, byte_ids):
        pad_mask = byte_ids != 256
        x = self.embed(byte_ids)
        if self.ngram_embed is not None:
            x = x + self.ngram_proj(self.ngram_embed(byte_ids))
        for layer in self.conv_layers:
            x = layer(x)
        for layer in self.attn_layers:
            x = layer(x)
        x = self.final_norm(x)
        mask = pad_mask.unsqueeze(-1).to(x.dtype)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(x)


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
