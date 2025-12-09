# MRI Brain Age Prediction

This project provides a complete pipeline for MRI brain age prediction using PyTorch. It loads volumetric MRI data stored as NumPy arrays, trains a regression model to predict brain age, and evaluates performance with common metrics.

The default backbone is a **3D U-Net** that outputs a voxel-wise age map and regresses scan-level age via spatial averaging. A ResNet-18 3D backbone remains available for comparison by switching a configuration flag.

## Data organization
```
data/
├── images/
│   ├── NC_116_S_4010.npy
│   └── ...
└── labels.csv  # columns: filename,age
```

Each `.npy` file should contain a single-channel volume with shape `(1, 113, 137, 113)`. The filename pattern encodes a subject ID and timepoint, e.g., `swm002_S_0295_0.npy`, `swm002_S_0295_1.npy`, where `swm002_S_0295` is the shared subject identifier.

On the first run, the training pipeline performs a **subject-wise split** (using the subject ID, not individual scans) to avoid timepoint leakage and writes:

```
data/
├── labels_train.csv
├── labels_val.csv
└── labels_test.csv
```

## Installation
```bash
pip install -r requirements.txt
```

## Training
```bash
python train.py
```

Set `BACKBONE` in `config.py` to `"unet"` (default) or `"resnet3d"`, and adjust `UNET_BASE_CHANNELS` / `DROPOUT` as needed.

## Evaluation
After training produces `checkpoints/best_model.pth`, evaluate on the held-out test set:
```bash
python evaluate.py
```

Metrics (MAE, RMSE, Pearson r) are logged to the console, and predictions are saved to `outputs/predictions.csv`.
