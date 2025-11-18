"""
Custom Dataset and DataLoader for 6G Bandwidth Allocation
Loads pre-normalized .npy sequence files and creates PyTorch DataLoaders
Data is already normalized during preprocessing - this just loads and provides scaler for denormalization
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
        # Load numpy arrays (already normalized during preprocessing)
        self.X = np.load(X_path)  # Shape: (num_samples, seq_len, features)
        self.y = np.load(y_path)  # Shape: (num_samples, output_size)
        
        # Verify shapes
        assert len(self.X) == len(self.y), \
            f"Mismatch: X has {len(self.X)} samples, y has {len(self.y)} samples"
        
        # Store scaler for denormalization during evaluation
        self.y_scaler = y_scaler
        
        print(f"✓ Loaded data from {X_path}")
        print(f"  - X shape: {self.X.shape}")
        print(f"  - y shape: {self.y.shape}")
        print(f"  - Data pre-normalized: YES (from preprocessing pipeline)")
        
        # Data statistics (should be normalized: mean≈0, std≈1)
        print(f"  - X range: [{self.X.min():.2f}, {self.X.max():.2f}]")
        print(f"  - y range: [{self.y.min():.2f}, {self.y.max():.2f}]")
    
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
    scaler_dir = os.path.dirname(config.X_TRAIN_PATH)
    target_scaler_path = os.path.join(scaler_dir, "target_scaler.pkl")
    
    if not os.path.exists(target_scaler_path):
        # Try alternate location (Processed folder)
        alt_scaler_path = "D:/Academics/SEM-5/Machine Learning/ML_courseproj/Processed/target_scaler.pkl"
        if os.path.exists(alt_scaler_path):
            target_scaler_path = alt_scaler_path
        else:
            raise FileNotFoundError(
                f"Target scaler not found at {target_scaler_path} or {alt_scaler_path}\n"
                f"Run preprocessing first: python preprocess_new_data.py"
            )
    
    target_scaler = joblib.load(target_scaler_path)
    print(f"\n✓ Loaded target scaler from: {target_scaler_path}")
    print(f"  - Scaler mean: {target_scaler.mean_}")
    print(f"  - Scaler std:  {target_scaler.scale_}")
    
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
    
    # Create DataLoaders
    print("\n" + "=" * 80)
    print("CREATING DATALOADERS")
    print("=" * 80)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,           # Shuffle training data
        num_workers=0,          # 0 for Windows, increase for Linux
        pin_memory=True         # Faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,          # Don't shuffle validation
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,          # Don't shuffle test
        num_workers=0,
        pin_memory=True
    )
    
    print(f"\n✓ DataLoaders created successfully!")
    print(f"  - Train: {len(train_loader.dataset):,} samples, {len(train_loader)} batches")
    print(f"  - Val:   {len(val_loader.dataset):,} samples, {len(val_loader)} batches")
    print(f"  - Test:  {len(test_loader.dataset):,} samples, {len(test_loader)} batches")
    print("=" * 80)
    
    return train_loader, val_loader, test_loader, target_scaler
