"""
Configuration file for 6G Bandwidth Allocation GRU Model
All hyperparameters and paths centralized here
Optimized for pre-normalized data with log-scale targets
"""

import torch
import os

# ===== DATA PATHS =====
DATA_DIR = r"D:\Academics\SEM-5\Machine Learning\ML_courseproj\SplitsMe"
X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train_lstm.npy")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train_lstm.npy")
X_VAL_PATH = os.path.join(DATA_DIR, "X_val_lstm.npy")
Y_VAL_PATH = os.path.join(DATA_DIR, "y_val_lstm.npy")
X_TEST_PATH = os.path.join(DATA_DIR, "X_test_lstm.npy")
Y_TEST_PATH = os.path.join(DATA_DIR, "y_test_lstm.npy")

# ===== PREPROCESSING PATHS =====
PROCESSED_DIR = r"D:\Academics\SEM-5\Machine Learning\ML_courseproj\Processed"
FEATURE_SCALER_PATH = os.path.join(PROCESSED_DIR, "feature_scaler.pkl")
TARGET_SCALER_PATH = os.path.join(PROCESSED_DIR, "target_scaler.pkl")

# ===== MODEL ARCHITECTURE =====
INPUT_SIZE = 17          # 17 features per timestep
HIDDEN_SIZE_1 = 128      # First GRU layer (increased capacity)
HIDDEN_SIZE_2 = 64       # Second GRU layer
DENSE_SIZE = 32          # Dense layer
OUTPUT_SIZE = 3          # 3 bandwidth predictions (eMBB, URLLC, mMTC)
DROPOUT = 0.3            # Dropout rate (increased for better regularization)

# ===== TRAINING HYPERPARAMETERS =====
BATCH_SIZE = 64          # Number of sequences per batch
LEARNING_RATE = 0.0005   # Adam optimizer learning rate (slightly increased)
NUM_EPOCHS = 100         # Maximum training epochs (increased for log-scale data)
RANDOM_SEED = 42         # For reproducibility
GRADIENT_CLIP_NORM = 1.0 # Gradient clipping threshold

# ===== OPTIMIZER SETTINGS =====
WEIGHT_DECAY = 1e-5      # L2 regularization (prevents overfitting)
LR_SCHEDULER_PATIENCE = 7  # ReduceLROnPlateau patience
LR_SCHEDULER_FACTOR = 0.5  # LR reduction factor
MIN_LR = 1e-7            # Minimum learning rate

# ===== DEVICE CONFIGURATION =====
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== MODEL SAVE PATH =====
PROJECT_ROOT = r"D:\Academics\SEM-5\Machine Learning\ML_courseproj"
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "GRU", "saved_models")
MODEL_NAME = "bandwidth_gru_best.pth"
FINAL_MODEL_NAME = "bandwidth_gru_final.pth"

# Create save directory if it doesn't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ===== LOSS WEIGHTS (for prioritizing URLLC) =====
# FIXED: Changed eMBB from 2.5 to 1.0 for balanced learning
LOSS_WEIGHTS = torch.tensor([1.0, 5.0, 1.0]).to(DEVICE)  # [eMBB, URLLC, mMTC]

# ===== EARLY STOPPING =====
EARLY_STOPPING_PATIENCE = 15  # Stop if no improvement after N epochs
EARLY_STOPPING_MIN_DELTA = 1e-4  # Minimum improvement to count as progress

# ===== LOGGING =====
LOG_INTERVAL = 5         # Evaluate detailed metrics every N epochs
SAVE_CHECKPOINT_INTERVAL = 0  # 0 = only save best model (no intermediate checkpoints)

# ===== DATA VALIDATION =====
# Verify paths exist
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(
        f"Data directory not found: {DATA_DIR}\n"
        f"Run sequence generation first: python create_sequences.py"
    )

for path in [X_TRAIN_PATH, Y_TRAIN_PATH, X_VAL_PATH, Y_VAL_PATH, X_TEST_PATH, Y_TEST_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            f"Run preprocessing and sequence generation first."
        )

# ===== PRINT CONFIGURATION =====
print("=" * 80)
print("6G BANDWIDTH ALLOCATION GRU - CONFIGURATION")
print("=" * 80)
print(f"\n DATA PATHS:")
print(f"  Data directory: {DATA_DIR}")
print(f"  Processed directory: {PROCESSED_DIR}")

print(f"\n MODEL ARCHITECTURE:")
print(f"  Input → GRU({HIDDEN_SIZE_1}) → GRU({HIDDEN_SIZE_2}) → Dense({DENSE_SIZE}) → Output({OUTPUT_SIZE})")
print(f"  Dropout: {DROPOUT}")
print(f"  Estimated parameters: ~100K")

print(f"\n TRAINING HYPERPARAMETERS:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Weight decay: {WEIGHT_DECAY}")
print(f"  Gradient clipping: {GRADIENT_CLIP_NORM}")

print(f"\n LOSS CONFIGURATION:")
print(f"  Weighted MSE Loss:")
print(f"    - eMBB weight:  {LOSS_WEIGHTS[0].item():.1f}")
print(f"    - URLLC weight: {LOSS_WEIGHTS[1].item():.1f} (prioritized)")
print(f"    - mMTC weight:  {LOSS_WEIGHTS[2].item():.1f}")

print(f"\n LEARNING RATE SCHEDULER:")
print(f"  Patience: {LR_SCHEDULER_PATIENCE} epochs")
print(f"  Factor: {LR_SCHEDULER_FACTOR}x reduction")
print(f"  Min LR: {MIN_LR}")

print(f"\n EARLY STOPPING:")
print(f"  Patience: {EARLY_STOPPING_PATIENCE} epochs")
print(f"  Min delta: {EARLY_STOPPING_MIN_DELTA}")

print(f"\n MODEL SAVING:")
print(f"  Save directory: {MODEL_SAVE_DIR}")
print(f"  Best model: {MODEL_NAME}")
print(f"  Final model: {FINAL_MODEL_NAME}")

print(f"\nCOMPUTE:")
print(f"  Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")

print("=" * 80 + "\n")
