"""Dataset utilities for MRI brain age prediction."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

from utils import log, set_random_seed


class BrainAgeDataset(Dataset):
    """Custom dataset for loading 3D MRI volumes and age labels."""

    def __init__(self, labels_csv: Path, images_dir: Path, augment: bool = False):
        self.labels_df = pd.read_csv(labels_csv)
        self.images_dir = images_dir
        self.augment = augment
        self.samples: List[Tuple[Path, float]] = [
            (images_dir / row["filename"], float(row["age"])) for _, row in self.labels_df.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, age = self.samples[idx]
        volume = np.load(image_path).astype(np.float32)
        tensor = torch.from_numpy(volume)
        tensor = self._standardize(tensor)

        if self.augment:
            tensor = self._random_flip(tensor)

        return tensor, torch.tensor(age, dtype=torch.float32)

    @staticmethod
    def _standardize(tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean()
        std = tensor.std().clamp(min=1e-6)
        return (tensor - mean) / std

    @staticmethod
    def _random_flip(tensor: torch.Tensor) -> torch.Tensor:
        for dim in [2, 3, 4]:
            if torch.rand(1).item() > 0.5:
                tensor = torch.flip(tensor, dims=(dim,))
        return tensor


def _extract_subject_id(filename: str) -> str:
    base = Path(filename).stem
    if "_" not in base:
        raise ValueError(f"Filename {filename} does not contain '_' to extract subject id")
    return base.rsplit("_", 1)[0]


def _ensure_subject_splits(cfg) -> None:
    """Create subject-wise train/val/test CSVs if they do not already exist."""

    target_files = [cfg.LABELS_TRAIN_CSV, cfg.LABELS_VAL_CSV, cfg.LABELS_TEST_CSV]
    if all(Path(p).exists() for p in target_files):
        log("Found existing subject-wise label splits; using them.")
        return

    df = pd.read_csv(cfg.LABELS_CSV)
    if "filename" not in df.columns or "age" not in df.columns:
        raise ValueError("labels.csv must contain 'filename' and 'age' columns")

    df = df.copy()
    df["subject_id"] = df["filename"].apply(_extract_subject_id)

    train_ratio, val_ratio, test_ratio = cfg.TRAIN_VAL_TEST_SPLIT
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("TRAIN_VAL_TEST_SPLIT should sum to 1.0")

    gss = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=cfg.RANDOM_SEED)
    train_idx, temp_idx = next(gss.split(df, groups=df["subject_id"]))

    temp_df = df.iloc[temp_idx].reset_index(drop=True)
    train_df = df.iloc[train_idx].reset_index(drop=True)

    remaining_ratio = val_ratio + test_ratio
    if remaining_ratio <= 0:
        raise ValueError("Validation and test ratios must be positive")
    val_relative = val_ratio / remaining_ratio

    gss_val = GroupShuffleSplit(n_splits=1, train_size=val_relative, random_state=cfg.RANDOM_SEED + 1)
    val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df["subject_id"]))

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    for path, split_df in zip(target_files, [train_df, val_df, test_df]):
        path.parent.mkdir(parents=True, exist_ok=True)
        split_df.drop(columns=["subject_id"]).to_csv(path, index=False)

    log(
        "Created subject-wise splits: "
        f"train={len(train_df)} val={len(val_df)} test={len(test_df)}"
    )


def create_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders using subject-wise splits."""
    set_random_seed(cfg.RANDOM_SEED)
    _ensure_subject_splits(cfg)

    train_dataset = BrainAgeDataset(cfg.LABELS_TRAIN_CSV, cfg.IMAGES_DIR, augment=True)
    val_dataset = BrainAgeDataset(cfg.LABELS_VAL_CSV, cfg.IMAGES_DIR, augment=False)
    test_dataset = BrainAgeDataset(cfg.LABELS_TEST_CSV, cfg.IMAGES_DIR, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=cfg.PERSISTENT_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=cfg.PERSISTENT_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=cfg.PERSISTENT_WORKERS,
    )

    return train_loader, val_loader, test_loader


__all__ = ["BrainAgeDataset", "create_dataloaders"]
