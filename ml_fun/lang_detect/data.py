import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, DistributedSampler

logger = logging.getLogger(__name__)

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
            logger.warning(
                f"Text {text[:10]} exceeds max length of {max_len} bytes and will be truncated."
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

    local_path = DATA_DIR / "commonlingua" / DATA_SPLITS[split]
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading {split} split from Hugging Face...")
        df = pl.read_parquet(
            "hf://datasets/PleIAs/CommonLingua-Train/" + DATA_SPLITS[split]
        )
        df.write_parquet(local_path)
        logger.info(f"Saved {split} split to cache at {local_path}")
    else:
        logger.info(f"Loading {split} split from cache...")
        df = pl.read_parquet(local_path)

    return df


class LangIdData(Dataset):
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


def setup_data_loaders(
    cfg: DictConfig, rank: int, world_size: int
) -> tuple[DistributedSampler, DataLoader, DistributedSampler, DataLoader]:
    train_data = LangIdData(load_data_cached("train"))
    train_sampler = DistributedSampler(
        train_data, num_replicas=world_size, shuffle=True, rank=rank
    )
    train_loader = DataLoader(
        train_data, batch_size=cfg.training.batch_size_train, sampler=train_sampler
    )

    eval_data = LangIdData(load_data_cached("validation"))
    eval_sampler = DistributedSampler(eval_data, num_replicas=world_size, rank=rank)
    eval_loader = DataLoader(
        eval_data, batch_size=cfg.training.batch_size_eval, sampler=eval_sampler
    )

    return train_sampler, train_loader, eval_sampler, eval_loader
