"""
Custom Dataset and DataLoader for 6G Bandwidth Allocation
Loads .npy sequence files and creates PyTorch DataLoaders
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BandwidthDataset(Dataset):
    """
    Custom PyTorch Dataset for loading .npy sequence files
    
    Args:
        X_path (str): Path to input sequences .npy file
        y_path (str): Path to target values .npy file
    """
    
    def __init__(self, X_path, y_path):
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
        tuple: (train_loader, val_loader, test_loader)
    """
    
    print("=" * 60)
    print("CREATING DATALOADERS")
    print("=" * 60)
    
    # Create datasets
    train_dataset = BandwidthDataset(config.X_TRAIN_PATH, config.Y_TRAIN_PATH)
    val_dataset = BandwidthDataset(config.X_VAL_PATH, config.Y_VAL_PATH)
    test_dataset = BandwidthDataset(config.X_TEST_PATH, config.Y_TEST_PATH)
    
    # Create DataLoaders
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
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    print("=" * 60)
    
    return train_loader, val_loader, test_loader
