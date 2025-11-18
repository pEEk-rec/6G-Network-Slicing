"""
Custom Dataset and DataLoader for 6G Bandwidth Allocation
Loads pre-normalized .npy sequence files and creates PyTorch DataLoaders
Includes data validation and quality checks
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import joblib
import os


class BandwidthDataset(Dataset):
    """
    Custom PyTorch Dataset for loading pre-normalized .npy sequence files
    
    Args:
        X_path (str): Path to input sequences .npy file (already normalized)
        y_path (str): Path to target values .npy file (already log-transformed & normalized)
        y_scaler: Pre-fitted scaler for inverse transform during evaluation (optional)
    """
    
    def __init__(self, X_path, y_path, y_scaler=None):
        """
        Initialize dataset by loading pre-normalized .npy files
        """
        # Verify files exist
        if not os.path.exists(X_path):
            raise FileNotFoundError(f"Input file not found: {X_path}")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"Target file not found: {y_path}")
        
        # Load numpy arrays (already normalized during preprocessing)
        self.X = np.load(X_path)  # Shape: (num_samples, seq_len, features)
        self.y = np.load(y_path)  # Shape: (num_samples, output_size)
        
        # Verify shapes
        assert len(self.X) == len(self.y), \
            f"Mismatch: X has {len(self.X)} samples, y has {len(self.y)} samples"
        
        assert self.X.ndim == 3, \
            f"Expected X to be 3D (samples, seq_len, features), got shape {self.X.shape}"
        
        assert self.y.ndim == 2, \
            f"Expected y to be 2D (samples, outputs), got shape {self.y.shape}"
        
        assert self.y.shape[1] == 3, \
            f"Expected y to have 3 outputs (eMBB, URLLC, mMTC), got {self.y.shape[1]}"
        
        # Store scaler for denormalization during evaluation
        self.y_scaler = y_scaler
        
        print(f"✓ Loaded data from {os.path.basename(X_path)}")
        print(f"  - X shape: {self.X.shape}")
        print(f"  - y shape: {self.y.shape}")
        print(f"  - Features per timestep: {self.X.shape[2]}")
        print(f"  - Sequence length: {self.X.shape[1]}")
        
        # Validate data quality
        self._validate_data()
        
        # Data statistics (should be normalized: mean≈0 for features, varies for targets)
        X_mean = self.X.mean()
        X_std = self.X.std()
        y_mean = self.y.mean()
        y_std = self.y.std()
        
        print(f"  - X stats: mean={X_mean:.3f}, std={X_std:.3f}")
        print(f"  - y stats: mean={y_mean:.3f}, std={y_std:.3f}")
        print(f"  - X range: [{self.X.min():.2f}, {self.X.max():.2f}]")
        print(f"  - y range: [{self.y.min():.2f}, {self.y.max():.2f}]")
        
        # Warn if data doesn't look normalized
        if abs(X_mean) > 1.0 or X_std < 0.5 or X_std > 2.0:
            print(f"  ⚠ WARNING: X statistics look unusual for normalized data")
            print(f"             Expected: mean≈0, std≈1")
            print(f"             Got: mean={X_mean:.3f}, std={X_std:.3f}")
    
    def _validate_data(self):
        """
        Validate data quality - check for NaN, Inf, and reasonable ranges
        """
        # Check for NaN
        nan_count_X = np.isnan(self.X).sum()
        nan_count_y = np.isnan(self.y).sum()
        
        if nan_count_X > 0:
            raise ValueError(f"Found {nan_count_X} NaN values in X data!")
        if nan_count_y > 0:
            raise ValueError(f"Found {nan_count_y} NaN values in y data!")
        
        # Check for Inf
        inf_count_X = np.isinf(self.X).sum()
        inf_count_y = np.isinf(self.y).sum()
        
        if inf_count_X > 0:
            raise ValueError(f"Found {inf_count_X} Inf values in X data!")
        if inf_count_y > 0:
            raise ValueError(f"Found {inf_count_y} Inf values in y data!")
        
        print(f"  ✓ Data quality check passed (no NaN/Inf)")
    
    def __len__(self):
        """
        Return total number of samples
        """
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Get a single sample (X, y) at index idx
        
        Args:
            idx (int): Index of sample to retrieve
        
        Returns:
            tuple: (input_sequence, target) as PyTorch tensors
        """
        # Convert numpy arrays to PyTorch tensors
        X_sample = torch.FloatTensor(self.X[idx])  # (seq_len, features)
        y_sample = torch.FloatTensor(self.y[idx])  # (output_size,)
        
        return X_sample, y_sample


def get_dataloaders(config):
    """
    Create train, validation, and test DataLoaders
    
    Args:
        config: Configuration module with paths and hyperparameters
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, target_scaler)
    """
    
    print("=" * 80)
    print("CREATING DATALOADERS (Pre-Normalized Data)")
    print("=" * 80)
    
    # Load the target scaler (saved during preprocessing)
    target_scaler_path = config.TARGET_SCALER_PATH
    
    if not os.path.exists(target_scaler_path):
        # Try alternate locations
        alt_paths = [
            os.path.join(os.path.dirname(config.X_TRAIN_PATH), "target_scaler.pkl"),
            "D:/Academics/SEM-5/Machine Learning/ML_courseproj/Processed/target_scaler.pkl"
        ]
        
        found = False
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                target_scaler_path = alt_path
                found = True
                break
        
        if not found:
            raise FileNotFoundError(
                f"Target scaler not found!\n"
                f"Searched locations:\n"
                f"  - {config.TARGET_SCALER_PATH}\n"
                f"  - {alt_paths[0]}\n"
                f"  - {alt_paths[1]}\n"
                f"\nRun preprocessing first: python preprocess_new_data.py"
            )
    
    target_scaler = joblib.load(target_scaler_path)
    print(f"\n✓ Loaded target scaler from: {target_scaler_path}")
    print(f"  - Scaler type: {type(target_scaler).__name__}")
    print(f"  - Scaler mean: {target_scaler.mean_}")
    print(f"  - Scaler std:  {target_scaler.scale_}")
    
    # Verify scaler has correct dimensions
    if len(target_scaler.mean_) != 3:
        raise ValueError(
            f"Target scaler has wrong dimensions!\n"
            f"Expected 3 (eMBB, URLLC, mMTC), got {len(target_scaler.mean_)}"
        )
    
    # Create datasets (data already normalized, just load)
    print("\n[1/3] Loading Training Data...")
    train_dataset = BandwidthDataset(
        config.X_TRAIN_PATH, 
        config.Y_TRAIN_PATH,
        y_scaler=target_scaler
    )
    
    print("\n[2/3] Loading Validation Data...")
    val_dataset = BandwidthDataset(
        config.X_VAL_PATH, 
        config.Y_VAL_PATH,
        y_scaler=target_scaler
    )
    
    print("\n[3/3] Loading Test Data...")
    test_dataset = BandwidthDataset(
        config.X_TEST_PATH, 
        config.Y_TEST_PATH,
        y_scaler=target_scaler
    )
    
    # Verify all datasets have same feature dimensions
    if train_dataset.X.shape[2] != val_dataset.X.shape[2] or \
       train_dataset.X.shape[2] != test_dataset.X.shape[2]:
        raise ValueError(
            f"Feature dimension mismatch!\n"
            f"Train: {train_dataset.X.shape[2]}, "
            f"Val: {val_dataset.X.shape[2]}, "
            f"Test: {test_dataset.X.shape[2]}"
        )
    
    # Verify feature count matches config
    expected_features = config.INPUT_SIZE
    actual_features = train_dataset.X.shape[2]
    
    if expected_features != actual_features:
        raise ValueError(
            f"Feature count mismatch!\n"
            f"Config INPUT_SIZE: {expected_features}\n"
            f"Actual features in data: {actual_features}\n"
            f"Update config.INPUT_SIZE to {actual_features}"
        )
    
    # Create DataLoaders
    print("\n" + "=" * 80)
    print("CREATING DATALOADERS")
    print("=" * 80)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\n✓ DataLoaders created successfully!")
    print(f"  - Train: {len(train_loader.dataset):,} samples, {len(train_loader)} batches")
    print(f"  - Val:   {len(val_loader.dataset):,} samples, {len(val_loader)} batches")
    print(f"  - Test:  {len(test_loader.dataset):,} samples, {len(test_loader)} batches")
    print(f"  - Batch size: {config.BATCH_SIZE}")
    print(f"  - Features per timestep: {actual_features}")
    print(f"  - Sequence length: {train_dataset.X.shape[1]}")
    print("=" * 80)
    
    return train_loader, val_loader, test_loader, target_scaler
