"""Training script for MRI brain age prediction."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import config
from dataset import create_dataloaders
from model import build_model
from utils import count_parameters, log, mae, pearsonr, rmse, set_random_seed


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        all_preds.extend(outputs.detach().cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            log(
                f"Batch {batch_idx + 1}/{len(loader)} - Loss: {loss.item():.4f} | "
                f"MAE: {mae([outputs.mean().item()], [targets.mean().item()]):.4f}"
            )

    epoch_loss = running_loss / len(loader.dataset)
    return {
        "loss": epoch_loss,
        "mae": mae(all_preds, all_targets),
        "rmse": rmse(all_preds, all_targets),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(outputs.detach().cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    return {
        "loss": epoch_loss,
        "mae": mae(all_preds, all_targets),
        "rmse": rmse(all_preds, all_targets),
        "pearson": pearsonr(all_preds, all_targets),
    }


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def main() -> None:
    set_random_seed(config.RANDOM_SEED)
    device = torch.device(config.DEVICE)
    log(f"Using device: {device}")

    train_loader, val_loader, _ = create_dataloaders(config)
    model = build_model(
        backbone=config.BACKBONE,
        dropout=config.DROPOUT,
        base_channels=config.UNET_BASE_CHANNELS,
        device=device,
    )
    total_params, trainable_params = count_parameters(model)
    log(f"Model parameters: total={total_params:,} trainable={trainable_params:,}")

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    best_val_loss = float("inf")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        log(f"Epoch {epoch}/{config.NUM_EPOCHS}")
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])

        log(
            " | ".join(
                [
                    f"Train - Loss: {train_metrics['loss']:.4f} MAE: {train_metrics['mae']:.4f} RMSE: {train_metrics['rmse']:.4f}",
                    f"Val - Loss: {val_metrics['loss']:.4f} MAE: {val_metrics['mae']:.4f} "
                    f"RMSE: {val_metrics['rmse']:.4f} r: {val_metrics['pearson']:.4f}",
                ]
            )
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(model, config.CHECKPOINT_DIR / "best_model.pth")
            log(f"Saved new best model with val loss {best_val_loss:.4f}")

    log("Training complete.")


if __name__ == "__main__":
    main()
