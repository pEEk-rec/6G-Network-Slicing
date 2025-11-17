"""
Configuration file for 6G Bandwidth Allocation GRU Model
All hyperparameters and paths centralized here
"""

import torch
import os

# ===== DATA PATHS =====
DATA_DIR = r"D:\Academics\SEM-5\Machine Learning\ML_courseproj\Splits"
X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train_lstm.npy")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train_lstm.npy")
X_VAL_PATH = os.path.join(DATA_DIR, "X_val_lstm.npy")
Y_VAL_PATH = os.path.join(DATA_DIR, "y_val_lstm.npy")
X_TEST_PATH = os.path.join(DATA_DIR, "X_test_lstm.npy")
Y_TEST_PATH = os.path.join(DATA_DIR, "y_test_lstm.npy")

# ===== MODEL ARCHITECTURE =====
INPUT_SIZE = 17          # 17 features per timestep
HIDDEN_SIZE_1 = 128       # First GRU layer
HIDDEN_SIZE_2 = 64      # Second GRU layer
DENSE_SIZE = 32       # Dense layer
OUTPUT_SIZE = 3          # 3 bandwidth predictions (eMBB, URLLC, mMTC)
DROPOUT = 0.2            # Dropout rate for regularization

# ===== TRAINING HYPERPARAMETERS =====
BATCH_SIZE = 64          # Number of sequences per batch
LEARNING_RATE = 0.0001    # Adam optimizer learning rate
NUM_EPOCHS = 50          # Maximum training epochs
RANDOM_SEED = 42         # For reproducibility

# ===== DEVICE CONFIGURATION =====
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== MODEL SAVE PATH =====
PROJECT_ROOT = r"D:\Academics\SEM-5\Machine Learning\ML_courseproj"
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "GRU", "saved_models")
MODEL_NAME = "bandwidth_gru_best.pth"

# Create save directory if it doesn't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ===== LOSS WEIGHTS (for prioritizing URLLC) =====
LOSS_WEIGHTS = torch.tensor([2.5, 5.0, 1.0]).to(DEVICE)  # [eMBB, URLLC, mMTC]

# ===== PRINT CONFIGURATION =====
print("=" * 60)
print("6G BANDWIDTH ALLOCATION GRU - CONFIGURATION")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Data directory: {DATA_DIR}")
print(f"Model save path: {os.path.join(MODEL_SAVE_DIR, MODEL_NAME)}")
print(f"Batch size: {BATCH_SIZE} | Epochs: {NUM_EPOCHS} | LR: {LEARNING_RATE}")
print("=" * 60)
