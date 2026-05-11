import os
import time

import aim
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ml_fun.lang_detect.data import DATA_DIR, IDX2LANG, LangIdData, load_data_cached
from ml_fun.lang_detect.metrics import accuracy, f1_score
from ml_fun.lang_detect.model import ByteHybrid


def setup() -> None:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def cleanup() -> None:
    dist.destroy_process_group()


def predict(
    net: nn.Module, data_loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[int] = []
    labels: list[int] = []
    for byte_ids, batch_labels in data_loader:
        with torch.no_grad():
            batch_logits = net(byte_ids.to(device))
            batch_preds = batch_logits.argmax(dim=1).cpu().tolist()
            predictions.extend(batch_preds)
            labels.extend(batch_labels.tolist())
    return np.asarray(predictions), np.asarray(labels)


def log_hparams(
    net: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    run: aim.Run,
) -> None:
    print(net)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    run["hparams"] = {
        "net": {
            "class": net.__class__.__name__,
            "num_classes": net.num_classes,
            "d_model": net.d_model,
            "n_conv": net.n_conv,
            "n_attn": net.n_attn,
            "n_heads": net.n_heads,
            "ffn_expand": net.ffn_expand,
            "conv_kernel": net.conv_kernel,
            "ngram_buckets": net.ngram_buckets,
            "ngram_dim": net.ngram_dim,
            "max_len": net.max_len,
            "num_epochs": num_epochs,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "num_params": num_params,
        },
        "optimizer": {
            "class": optimizer.__class__.__name__,
            "learning_rate": optimizer.param_groups[0]["lr"],
        },
        "misc": {
            "device": str(device),
            "epochs": num_epochs,
        },
    }
    print(f"Total parameters: {num_params:_}")


def train() -> None:
    setup()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")

    print(f"Rank {rank}/{world_size} running on {device}")

    run = aim.Run(experiment="lang_detect")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print(f"Starting training at {timestamp}")
    num_epochs = 10
    num_evals_per_epoch = 10
    print(f"Using device: {device}")
    net = ByteHybrid(
        num_classes=len(IDX2LANG),
        d_model=256,
        n_conv=3,
        n_attn=1,
        n_heads=4,
        ffn_expand=2,
        conv_kernel=15,
        ngram_buckets=4096,
        ngram_dim=64,
        max_len=512,
    ).to(device)
    ddp_net = DDP(net, device_ids=[device.index] if device.type == "cuda" else None)
    optimizer = optim.Adam(ddp_net.parameters(), lr=1e-3)
    train_data = LangIdData(load_data_cached("train"))
    train_sampler = DistributedSampler(
        train_data, num_replicas=world_size, shuffle=True, rank=rank
    )
    train_loader = DataLoader(train_data, batch_size=128, sampler=train_sampler)

    eval_data = LangIdData(load_data_cached("validation"))
    eval_sampler = DistributedSampler(eval_data, num_replicas=world_size, rank=rank)
    eval_loader = DataLoader(eval_data, batch_size=128, sampler=eval_sampler)

    log_hparams(net, optimizer, device, num_epochs, run)
    global_step = 1
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        net.train()
        total_loss = 0.0
        p_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        num_steps = len(train_loader)
        for b_idx, (byte_ids, labels) in enumerate(p_bar):
            global_step += 1
            optimizer.zero_grad()
            logits = net(byte_ids.to(device))
            loss = F.cross_entropy(logits, labels.to(device))
            loss.backward()
            run.track(loss.item(), name="loss", epoch=epoch, step=b_idx)
            optimizer.step()
            total_loss += loss.item()
            if (b_idx + 1) % (num_steps // num_evals_per_epoch) == 0:
                print(f"Batch {b_idx+1} Loss: {loss.item():.4f}")
                print("Starting evaluation...")
                net.eval()
                predictions, labels = predict(net, eval_loader, device)
                net.train()
                accuracy_score = accuracy(predictions, labels)
                run.track(
                    accuracy_score, name="accuracy", epoch=epoch, step=global_step
                )
                print(f"Accuracy: {100.* accuracy_score:.4f}%")
                macro_f1 = sum(
                    f1_score(predictions, labels, class_id=lang_id)
                    for lang_id in range(len(IDX2LANG))
                ) / len(IDX2LANG)
                run.track(macro_f1, name="macro_f1", epoch=epoch, step=global_step)
                print(f"Macro F1 Score: {macro_f1:.4f}")
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        print("Starting evaluation...")
        net.eval()
        predictions, labels = predict(net, eval_loader, device)
        net.train()
        print(f"Accuracy: {100.* accuracy(predictions, labels):.4f}%")
        macro_f1 = sum(
            f1_score(predictions, labels, class_id=lang_id)
            for lang_id in range(len(IDX2LANG))
        ) / len(IDX2LANG)
        print(f"Macro F1 Score: {macro_f1:.4f}")

        print(f"Saving checkpoint for epoch {epoch+1}...")
        (DATA_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            DATA_DIR / "checkpoints" / f"byte_hybrid_epoch_{timestamp}_{epoch+1}.pt",
        )
    cleanup()


if __name__ == "__main__":
    train()
