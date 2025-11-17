"""
Training script for 6G Bandwidth Allocation GRU Model
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from datetime import datetime

from config import config
from data.dataloader import get_dataloaders
from models.gru_model import BandwidthAllocationGRU
from utils.helpers import save_checkpoint, count_parameters


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch
    
    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        # Move data to device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] - Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, criterion, device):
    """
    Validate model
    
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def main():
    """
    Main training function
    """
    print("\n" + "=" * 70)
    print("TRAINING 6G BANDWIDTH ALLOCATION GRU MODEL")
    print("=" * 70)
    
    # Set random seed for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # Initialize model
    print("\n" + "=" * 70)
    print("INITIALIZING MODEL")
    print("=" * 70)
    model = BandwidthAllocationGRU(
        input_size=config.INPUT_SIZE,
        hidden_size_1=config.HIDDEN_SIZE_1,
        hidden_size_2=config.HIDDEN_SIZE_2,
        dense_size=config.DENSE_SIZE,
        output_size=config.OUTPUT_SIZE,
        dropout=config.DROPOUT
    )
    model = model.to(config.DEVICE)
    model.print_architecture()
    
    # Define loss function with weights for URLLC priority
    criterion = nn.MSELoss(reduction='none')
    
    def weighted_mse_loss(predictions, targets):
        """Custom weighted MSE loss"""
        losses = criterion(predictions, targets)  # (batch, 3)
        weighted_losses = losses * config.LOSS_WEIGHTS
        return weighted_losses.mean()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training tracking
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Create TensorBoard writer (optional)
    log_dir = os.path.join(config.MODEL_SAVE_DIR, "logs", 
                           datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}]")
        print("-" * 70)
        
        # Train
        train_loss = train_one_epoch(model, train_loader, weighted_mse_loss, 
                                     optimizer, config.DEVICE)
        
        # Validate
        val_loss = validate(model, val_loader, weighted_mse_loss, config.DEVICE)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"\n  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(config.MODEL_SAVE_DIR, config.MODEL_NAME)
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, 
                          best_model_path)
            print(f"  ✓ New best model! Val Loss: {val_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, 
                                          f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, 
                          checkpoint_path)
    
    writer.close()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model saved at: {os.path.join(config.MODEL_SAVE_DIR, config.MODEL_NAME)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
