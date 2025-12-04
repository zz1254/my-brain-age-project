# MRI Brain Age Prediction

This project provides a complete pipeline for MRI brain age prediction using a 3D ResNet-18 model built with PyTorch. It loads volumetric MRI data stored as NumPy arrays, trains a regression model to predict brain age, and evaluates performance with common metrics.

## Data organization
```
data/
├── images/
│   ├── NC_116_S_4010.npy
│   └── ...
└── labels.csv  # columns: filename,age
```

Each `.npy` file should contain a single-channel volume with shape `(1, 113, 137, 113)`.

## Installation
```bash
pip install -r requirements.txt
```

## Training
```bash
python train.py
```

## Evaluation
After training produces `checkpoints/best_model.pth`, evaluate on the held-out test set:
```bash
python evaluate.py
```

Metrics (MAE, RMSE, Pearson r) are logged to the console, and predictions are saved to `outputs/predictions.csv`.
