"""
Hierarchical GRU Model for 6G Bandwidth Allocation
Predicts bandwidth allocation for eMBB, URLLC, and mMTC slices
"""

import torch
import torch.nn as nn


class BandwidthAllocationGRU(nn.Module):
    """
    Hierarchical GRU for 6G Network Slice Bandwidth Allocation
    
    Architecture:
    - Input Layer: 17 features per timestep
    - GRU Layer 1: 128 hidden units (captures immediate temporal patterns)
    - GRU Layer 2: 64 hidden units (captures higher-level trends)
    - Batch Normalization: Stabilizes training
    - Dense Layer: 32 units (non-linear mapping)
    - Output Layer: 3 units (bandwidth for eMBB, URLLC, mMTC)
    
    Args:
        input_size (int): Number of input features per timestep (default: 17)
        hidden_size_1 (int): Hidden units in first GRU layer (default: 128)
        hidden_size_2 (int): Hidden units in second GRU layer (default: 64)
        dense_size (int): Units in dense layer (default: 32)
        output_size (int): Number of output predictions (default: 3)
        dropout (float): Dropout rate for regularization (default: 0.2)
    """
    
    def __init__(self, input_size=17, hidden_size_1=128, hidden_size_2=64, 
                 dense_size=32, output_size=3, dropout=0.2):
        super(BandwidthAllocationGRU, self).__init__()
        
        # Store architecture parameters
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.dense_size = dense_size
        self.output_size = output_size
        self.dropout_rate = dropout
        
        # === Layer 1: First GRU (128 units) ===
        self.gru1 = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size_1,
            num_layers=1,
            batch_first=True,
            dropout=0  # No dropout for single layer
        )
        
        # === Layer 2: Second GRU (64 units) ===
        self.gru2 = nn.GRU(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # === Batch Normalization (stabilizes training) ===
        self.bn1 = nn.BatchNorm1d(hidden_size_2)
        
        # === Dropout for regularization ===
        self.dropout = nn.Dropout(dropout)
        
        # === Layer 3: Dense layer (32 units) ===
        self.fc1 = nn.Linear(hidden_size_2, dense_size)
        self.bn2 = nn.BatchNorm1d(dense_size)
        self.relu = nn.ReLU()
        
        # === Layer 4: Output layer (3 units) ===
        self.fc_out = nn.Linear(dense_size, output_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize weights using Xavier uniform initialization
        Better convergence for deep networks
        """
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Hidden-hidden weights (recurrent)
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # Biases
                nn.init.zeros_(param)
            elif 'fc' in name and 'weight' in name:
                # Fully connected weights
                nn.init.xavier_uniform_(param)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, seq_len=10, features=17)
        
        Returns:
            Predicted bandwidth allocations (batch_size, 3)
        """
        batch_size = x.size(0)
        
        # === GRU Layer 1 ===
        # Input: (batch, 10, 17)
        # Output: (batch, 10, 128)
        out, _ = self.gru1(x)
        
        # === GRU Layer 2 ===
        # Input: (batch, 10, 128)
        # Output: (batch, 10, 64)
        out, hidden = self.gru2(out)
        
        # === Extract final timestep ===
        # Take only last timestep output
        # Shape: (batch, 64)
        out = out[:, -1, :]
        
        # === Batch Normalization ===
        out = self.bn1(out)
        
        # === Apply dropout ===
        out = self.dropout(out)
        
        # === Dense layer ===
        # Input: (batch, 64)
        # Output: (batch, 32)
        out = self.fc1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # === Output layer ===
        # Input: (batch, 32)
        # Output: (batch, 3)
        out = self.fc_out(out)
        
        return out
    
    def get_num_parameters(self):
        """
        Calculate total number of trainable parameters
        
        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_architecture(self):
        """
        Print detailed model architecture
        """
        print("=" * 80)
        print("MODEL ARCHITECTURE: BandwidthAllocationGRU (Enhanced)")
        print("=" * 80)
        print(f"Input size:          {self.input_size} features")
        print(f"GRU Layer 1:         {self.hidden_size_1} hidden units")
        print(f"GRU Layer 2:         {self.hidden_size_2} hidden units")
        print(f"Batch Norm:          Enabled (after GRU2 and Dense)")
        print(f"Dense Layer:         {self.dense_size} units")
        print(f"Dropout rate:        {self.dropout_rate}")
        print(f"Output size:         {self.output_size} predictions (eMBB, URLLC, mMTC)")
        print(f"\nTotal parameters:    {self.get_num_parameters():,}")
        print(f"Weight init:         Xavier Uniform (FC), Orthogonal (Recurrent)")
        print("=" * 80)


# For testing the model
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TESTING GRU MODEL")
    print("=" * 80)
    
    # Create model instance
    model = BandwidthAllocationGRU(
        input_size=17,
        hidden_size_1=128,
        hidden_size_2=64,
        dense_size=32,
        output_size=3,
        dropout=0.2
    )
    
    # Print architecture
    model.print_architecture()
    
    # Test forward pass
    print("\n" + "=" * 80)
    print("TESTING FORWARD PASS")
    print("=" * 80)
    
    batch_size = 32
    seq_len = 10
    n_features = 17
    
    # Create random input
    dummy_input = torch.randn(batch_size, seq_len, n_features)
    print(f"Input shape:  {dummy_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"\nSample predictions (first 3 samples):")
    print(output[:3])
    
    print("\n" + "=" * 80)
    print("✓ Model test passed!")
    print("=" * 80 + "\n")
