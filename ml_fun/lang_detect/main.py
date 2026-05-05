import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml_fun.lang_detect.data import DATA_DIR, IDX2LANG, LangIdData, load_data_cached
from ml_fun.lang_detect.model import ByteHybrid


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


def train() -> None:
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
    train_loader = DataLoader(LangIdData(df), batch_size=128, shuffle=True)
    eval_loader = DataLoader(LangIdData(load_data_cached("validation")), batch_size=128)

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


if __name__ == "__main__":
    train()
