"""
Custom Dataset and DataLoader for 6G Bandwidth Allocation
Loads .npy sequence files and creates PyTorch DataLoaders with target normalization
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import os


class BandwidthDataset(Dataset):
    """
    Custom PyTorch Dataset for loading .npy sequence files
    
    Args:
        X_path (str): Path to input sequences .npy file
        y_path (str): Path to target values .npy file
        y_scaler (StandardScaler, optional): Pre-fitted scaler for targets
        fit_scaler (bool): If True, fit a new scaler on this data
    """
    
    def __init__(self, X_path, y_path, y_scaler=None, fit_scaler=False):
        """
        Initialize dataset by loading .npy files
        """
        # Load numpy arrays
        self.X = np.load(X_path)  # Shape: (num_samples, seq_len, features)
        self.y = np.load(y_path)  # Shape: (num_samples, output_size)
        
        # Verify shapes
        assert len(self.X) == len(self.y), \
            f"Mismatch: X has {len(self.X)} samples, y has {len(self.y)} samples"
        
        print(f"✓ Loaded data from {X_path}")
        print(f"  - X shape: {self.X.shape}")
        print(f"  - y shape: {self.y.shape}")
        
        # Handle target normalization
        if fit_scaler:
            # Fit new scaler on training data
            self.y_scaler = StandardScaler()
            self.y = self.y_scaler.fit_transform(self.y)
            print(f"  - y scaler fitted (NEW)")
            print(f"    Mean: {self.y_scaler.mean_}")
            print(f"    Std:  {self.y_scaler.scale_}")
        elif y_scaler is not None:
            # Use existing scaler (for val/test)
            self.y_scaler = y_scaler
            self.y = self.y_scaler.transform(self.y)
            print(f"  - y normalized (using train scaler)")
        else:
            # No normalization
            self.y_scaler = None
            print(f"  - y NOT normalized (WARNING: may cause training issues!)")
    
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
    Create train, validation, and test DataLoaders with proper target normalization
    
    Args:
        config: Configuration module with paths and hyperparameters
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, target_scaler)
    """
    
    print("=" * 80)
    print("CREATING DATALOADERS WITH TARGET NORMALIZATION")
    print("=" * 80)
    
    # Create train dataset and FIT scaler on training targets
    print("\n[1/3] Loading Training Data...")
    train_dataset = BandwidthDataset(
        config.X_TRAIN_PATH, 
        config.Y_TRAIN_PATH,
        fit_scaler=True  # Fit scaler on training data
    )
    
    # Create val dataset using TRAIN scaler (prevent data leakage)
    print("\n[2/3] Loading Validation Data...")
    val_dataset = BandwidthDataset(
        config.X_VAL_PATH, 
        config.Y_VAL_PATH,
        y_scaler=train_dataset.y_scaler  # Use train scaler
    )
    
    # Create test dataset using TRAIN scaler
    print("\n[3/3] Loading Test Data...")
    test_dataset = BandwidthDataset(
        config.X_TEST_PATH, 
        config.Y_TEST_PATH,
        y_scaler=train_dataset.y_scaler  # Use train scaler
    )
    
    # Save target scaler for inference
    scaler_save_path = os.path.join(os.path.dirname(config.X_TRAIN_PATH), 
                                    "target_scaler.pkl")
    joblib.dump(train_dataset.y_scaler, scaler_save_path)
    print(f"\n✓ Target scaler saved: {scaler_save_path}")
    
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
    
    return train_loader, val_loader, test_loader, train_dataset.y_scaler
