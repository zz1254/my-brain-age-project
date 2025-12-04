"""Evaluation script for MRI brain age prediction."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn

import config
from dataset import create_dataloaders
from model import build_model
from utils import log, mae, pearsonr, rmse, set_random_seed


def run_inference(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_preds: List[float] = []
    all_targets: List[float] = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().tolist())
            all_targets.extend(targets.tolist())

    return {
        "mae": mae(all_preds, all_targets),
        "rmse": rmse(all_preds, all_targets),
        "pearson": pearsonr(all_preds, all_targets),
        "preds": all_preds,
        "targets": all_targets,
    }


def save_predictions(preds: List[float], targets: List[float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"target": targets, "prediction": preds})
    df.to_csv(path, index=False)


def main() -> None:
    set_random_seed(config.RANDOM_SEED)
    device = torch.device(config.DEVICE)
    log(f"Using device: {device}")

    _, _, test_loader = create_dataloaders(config)
    model = build_model(device=device)

    checkpoint_path = config.CHECKPOINT_DIR / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    log("Loaded checkpoint.")

    metrics = run_inference(model, test_loader, device)
    log(
        f"Test - MAE: {metrics['mae']:.4f} RMSE: {metrics['rmse']:.4f} "
        f"r: {metrics['pearson']:.4f}"
    )

    save_predictions(metrics["preds"], metrics["targets"], config.OUTPUT_DIR / "predictions.csv")
    log(f"Saved predictions to {config.OUTPUT_DIR / 'predictions.csv'}")


if __name__ == "__main__":
    main()
