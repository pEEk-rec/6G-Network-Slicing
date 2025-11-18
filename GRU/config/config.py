"""
Configuration file for 6G Bandwidth Allocation GRU Model
FIXED VERSION - Addresses catastrophic loss weighting issue
Updated for improved preprocessing with 39 features
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

# ===== MODEL ARCHITECTURE ===== (INCREASED CAPACITY)
INPUT_SIZE = 39          # Updated from 17 to 39 (new feature engineering)
HIDDEN_SIZE_1 = 256      # Doubled from 128 (more capacity)
HIDDEN_SIZE_2 = 128      # Doubled from 64
HIDDEN_SIZE_3 = 64       # NEW third layer
DENSE_SIZE = 64          # Doubled from 32
OUTPUT_SIZE = 3          # 3 bandwidth predictions (eMBB, URLLC, mMTC)
DROPOUT = 0.3            # Kept at 0.3

# ===== TRAINING HYPERPARAMETERS =====
BATCH_SIZE = 64          
LEARNING_RATE = 0.001    # Increased from 0.0005 for faster convergence
NUM_EPOCHS = 100         
RANDOM_SEED = 42         
GRADIENT_CLIP_NORM = 1.0 

# ===== OPTIMIZER SETTINGS =====
WEIGHT_DECAY = 1e-5      # L2 regularization
LR_SCHEDULER_PATIENCE = 7  
LR_SCHEDULER_FACTOR = 0.5  
MIN_LR = 1e-7            

# ===== DEVICE CONFIGURATION =====
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== MODEL SAVE PATH =====
PROJECT_ROOT = r"D:\Academics\SEM-5\Machine Learning\ML_courseproj"
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "GRU", "saved_models")
MODEL_NAME = "bandwidth_gru_best.pth"
FINAL_MODEL_NAME = "bandwidth_gru_final.pth"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ===== LOSS CONFIGURATION ===== (CRITICAL FIX)
# REMOVED HARMFUL URLLC WEIGHTING - Log-normalization handles scale differences
LOSS_WEIGHTS = torch.tensor([1.0, 1.0, 1.0]).to(DEVICE)  # Equal weights
USE_MAE_LOSS = False  # Set True to use MAE instead of MSE
TRACK_PER_SLICE_LOSS = True  # Monitor individual slice losses during training

# ===== EARLY STOPPING =====
EARLY_STOPPING_PATIENCE = 15  
EARLY_STOPPING_MIN_DELTA = 1e-4  

# ===== LOGGING =====
LOG_INTERVAL = 5         
SAVE_CHECKPOINT_INTERVAL = 0  

# ===== DATA VALIDATION =====
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
print("6G BANDWIDTH ALLOCATION GRU - FIXED CONFIGURATION")
print("=" * 80)

print(f"\nDATA PATHS:")
print(f"  Data directory: {DATA_DIR}")
print(f"  Processed directory: {PROCESSED_DIR}")

print(f"\nMODEL ARCHITECTURE (INCREASED CAPACITY):")
print(f"  Input({INPUT_SIZE}) -> GRU({HIDDEN_SIZE_1}) -> GRU({HIDDEN_SIZE_2}) -> GRU({HIDDEN_SIZE_3}) -> Dense({DENSE_SIZE}) -> Output({OUTPUT_SIZE})")
print(f"  Dropout: {DROPOUT}")
print(f"  Estimated parameters: ~400K (4x increase)")

print(f"\nTRAINING HYPERPARAMETERS:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE} (increased)")
print(f"  Weight decay: {WEIGHT_DECAY}")
print(f"  Gradient clipping: {GRADIENT_CLIP_NORM}")

print(f"\nLOSS CONFIGURATION (CRITICAL FIX):")
print(f"  Loss function: {'MAE' if USE_MAE_LOSS else 'MSE'}")
print(f"  Loss weights: EQUAL [1.0, 1.0, 1.0]")
print(f"  Rationale: Log-normalization already handles scale differences")
print(f"  Per-slice tracking: {TRACK_PER_SLICE_LOSS}")
print(f"  REMOVED: Harmful 5x URLLC weighting that caused optimization collapse")

print(f"\nLEARNING RATE SCHEDULER:")
print(f"  Patience: {LR_SCHEDULER_PATIENCE} epochs")
print(f"  Factor: {LR_SCHEDULER_FACTOR}x reduction")
print(f"  Min LR: {MIN_LR}")

print(f"\nEARLY STOPPING:")
print(f"  Patience: {EARLY_STOPPING_PATIENCE} epochs")
print(f"  Min delta: {EARLY_STOPPING_MIN_DELTA}")

print(f"\nMODEL SAVING:")
print(f"  Save directory: {MODEL_SAVE_DIR}")
print(f"  Best model: {MODEL_NAME}")
print(f"  Final model: {FINAL_MODEL_NAME}")

print(f"\nCOMPUTE:")
print(f"  Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")

print("\n" + "=" * 80)
print("KEY FIXES APPLIED:")
print("-" * 80)
print("1. LOSS WEIGHTS: Changed from [1.0, 5.0, 1.0] to [1.0, 1.0, 1.0]")
print("   - Previous 5x URLLC weight caused model to ignore eMBB and mMTC")
print("   - Log-normalization already equalizes scales")
print("2. MODEL CAPACITY: 2 layers -> 3 layers, doubled hidden sizes")
print("   - 100K params -> 400K params for better pattern learning")
print("3. INPUT FEATURES: 17 -> 39 (improved feature engineering)")
print("   - Added demand ratios, temporal deltas, channel quality")
print("4. LEARNING RATE: 0.0005 -> 0.001 (faster initial convergence)")
print("5. PER-SLICE TRACKING: Enabled to monitor individual slice optimization")
print("=" * 80 + "\n")
