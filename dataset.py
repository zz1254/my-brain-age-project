"""Dataset utilities for MRI brain age prediction."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from utils import set_random_seed


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


def create_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders using provided configuration."""
    set_random_seed(cfg.RANDOM_SEED)
    dataset = BrainAgeDataset(cfg.LABELS_CSV, cfg.IMAGES_DIR, augment=True)
    total_size = len(dataset)

    train_ratio, val_ratio, test_ratio = cfg.TRAIN_VAL_TEST_SPLIT
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("TRAIN_VAL_TEST_SPLIT should sum to 1.0")

    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(cfg.RANDOM_SEED),
    )

    # Disable augmentation for validation and test sets
    val_dataset.dataset.augment = False
    test_dataset.dataset.augment = False

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
