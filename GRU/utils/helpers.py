"""
Helper functions for training and evaluation
"""

import torch
import os
from datetime import datetime


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, filepath):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        filepath: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        tuple: (model, optimizer, epoch, train_loss, val_loss)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Checkpoint loaded from: {filepath}")
    print(f"  - Epoch: {checkpoint['epoch']}")
    print(f"  - Train Loss: {checkpoint['train_loss']:.4f}")
    print(f"  - Val Loss: {checkpoint['val_loss']:.4f}")
    
    return model, optimizer, checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']


def count_parameters(model):
    """
    Count trainable parameters
    
    Args:
        model: PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
