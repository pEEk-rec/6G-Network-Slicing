"""
Test script to verify DataLoader is working correctly
"""

import sys
sys.path.append('.')

from data.dataloader import get_dataloaders
from config import config

# Create dataloaders
train_loader, val_loader, test_loader = get_dataloaders(config)

# Test: Get one batch from training data
print("\n" + "=" * 60)
print("TESTING DATA LOADING")
print("=" * 60)

for X_batch, y_batch in train_loader:
    print(f"✓ Successfully loaded one batch:")
    print(f"  - X_batch shape: {X_batch.shape}")  # Should be (batch_size, 10, 17)
    print(f"  - y_batch shape: {y_batch.shape}")  # Should be (batch_size, 3)
    print(f"  - X_batch dtype: {X_batch.dtype}")
    print(f"  - y_batch dtype: {y_batch.dtype}")
    print(f"\n  Sample input (first sequence, first timestep, first 5 features):")
    print(f"  {X_batch[0, 0, :5]}")
    print(f"\n  Sample target (first sample):")
    print(f"  {y_batch[0]}")
    break  # Only test one batch

print("=" * 60)
print("✓ DataLoader test passed!")
print("=" * 60)
