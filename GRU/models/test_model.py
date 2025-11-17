"""
Test script to verify GRU model is working correctly
"""

import sys
sys.path.append('.')

import torch
from models.gru_model import BandwidthAllocationGRU
from config import config

print("=" * 70)
print("TESTING GRU MODEL")
print("=" * 70)

# Create model instance
model = BandwidthAllocationGRU(
    input_size=config.INPUT_SIZE,
    hidden_size_1=config.HIDDEN_SIZE_1,
    hidden_size_2=config.HIDDEN_SIZE_2,
    dense_size=config.DENSE_SIZE,
    output_size=config.OUTPUT_SIZE,
    dropout=config.DROPOUT
)

# Print architecture
model.print_architecture()

# Move model to device
model = model.to(config.DEVICE)
print(f"\n✓ Model moved to: {config.DEVICE}")

# Test with dummy input
print("\n" + "=" * 70)
print("TESTING FORWARD PASS")
print("=" * 70)

batch_size = 32
seq_len = 10
n_features = 17

# Create random input
dummy_input = torch.randn(batch_size, seq_len, n_features).to(config.DEVICE)
print(f"Input shape: {dummy_input.shape}")

# Forward pass
model.eval()
with torch.no_grad():
    output = model(dummy_input)

print(f"Output shape: {output.shape}")
print(f"\nSample predictions (first 3 samples):")
print(output[:3])

print("\n" + "=" * 70)
print("✓ Model test passed!")
print("=" * 70)
