"""Configuration for MRI brain age prediction project."""
from pathlib import Path
import torch

# Base paths
DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
LABELS_CSV = DATA_DIR / "labels.csv"
LABELS_TRAIN_CSV = DATA_DIR / "labels_train.csv"
LABELS_VAL_CSV = DATA_DIR / "labels_val.csv"
LABELS_TEST_CSV = DATA_DIR / "labels_test.csv"
CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("outputs")

# Training hyperparameters
BATCH_SIZE = 2
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
TRAIN_VAL_TEST_SPLIT = (0.7, 0.15, 0.15)
RANDOM_SEED = 42

# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Logging
LOG_INTERVAL = 10

# Misc
PIN_MEMORY = True
PERSISTENT_WORKERS = False
