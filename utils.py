"""Utility functions for training and evaluation."""
from __future__ import annotations

import datetime as dt
import random
from typing import Iterable, Tuple

import numpy as np
import torch


def mae(preds: Iterable[float], targets: Iterable[float]) -> float:
    preds_arr = np.asarray(list(preds), dtype=np.float32)
    targets_arr = np.asarray(list(targets), dtype=np.float32)
    return float(np.mean(np.abs(preds_arr - targets_arr)))


def rmse(preds: Iterable[float], targets: Iterable[float]) -> float:
    preds_arr = np.asarray(list(preds), dtype=np.float32)
    targets_arr = np.asarray(list(targets), dtype=np.float32)
    return float(np.sqrt(np.mean((preds_arr - targets_arr) ** 2)))


def pearsonr(preds: Iterable[float], targets: Iterable[float]) -> float:
    preds_arr = np.asarray(list(preds), dtype=np.float32)
    targets_arr = np.asarray(list(targets), dtype=np.float32)
    if preds_arr.size == 0 or targets_arr.size == 0:
        return float("nan")
    preds_centered = preds_arr - preds_arr.mean()
    targets_centered = targets_arr - targets_arr.mean()
    numerator = np.sum(preds_centered * targets_centered)
    denominator = np.sqrt(np.sum(preds_centered**2) * np.sum(targets_centered**2))
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def log(message: str) -> None:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


__all__ = ["mae", "rmse", "pearsonr", "log", "set_random_seed", "count_parameters"]
