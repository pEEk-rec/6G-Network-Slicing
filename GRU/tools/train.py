"""
Training script for 6G Bandwidth Allocation GRU Model
MMAction2-style training logs with comprehensive metrics
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from datetime import datetime
from tqdm import tqdm

from config import config
from data.dataloader import get_dataloaders
from models.gru_model import BandwidthAllocationGRU
from utils.helpers import save_checkpoint, count_parameters
from utils.metrics import evaluate_model, print_metrics


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    """
    Train model for one epoch with MMAction2-style logging
    
    Returns:
        float: Average training loss
    """
    model.train()
    loss_meter = AverageMeter()
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch [{epoch}/{num_epochs}] Train', 
                ncols=120, ascii=True)
    
    for batch_idx, (X_batch, y_batch) in enumerate(pbar):
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
        
        # Update metrics
        loss_meter.update(loss.item(), X_batch.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.val:.4f}',
            'avg_loss': f'{loss_meter.avg:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    return loss_meter.avg


def validate(model, val_loader, criterion, device, epoch, num_epochs):
    """
    Validate model with MMAction2-style logging
    
    Returns:
        float: Average validation loss
    """
    model.eval()
    loss_meter = AverageMeter()
    
    # Progress bar
    pbar = tqdm(val_loader, desc=f'Epoch [{epoch}/{num_epochs}] Val  ', 
                ncols=120, ascii=True)
    
    with torch.no_grad():
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            # Update metrics
            loss_meter.update(loss.item(), X_batch.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.val:.4f}',
                'avg_loss': f'{loss_meter.avg:.4f}'
            })
    
    return loss_meter.avg


def main():
    """
    Main training function
    """
    # Print header
    print("\n" + "=" * 80)
    print(" " * 20 + "6G BANDWIDTH ALLOCATION GRU TRAINING")
    print("=" * 80)
    
    # Set random seed for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # Initialize model
    print("\n" + "-" * 80)
    print("MODEL INITIALIZATION")
    print("-" * 80)
    model = BandwidthAllocationGRU(
        input_size=config.INPUT_SIZE,
        hidden_size_1=config.HIDDEN_SIZE_1,
        hidden_size_2=config.HIDDEN_SIZE_2,
        dense_size=config.DENSE_SIZE,
        output_size=config.OUTPUT_SIZE,
        dropout=config.DROPOUT
    )
    model = model.to(config.DEVICE)
    
    # Print model info
    num_params = count_parameters(model)
    print(f"Model: BandwidthAllocationGRU")
    print(f"Architecture: Input({config.INPUT_SIZE}) -> GRU({config.HIDDEN_SIZE_1}) -> "
          f"GRU({config.HIDDEN_SIZE_2}) -> Dense({config.DENSE_SIZE}) -> Output({config.OUTPUT_SIZE})")
    print(f"Total params: {num_params:,}")
    print(f"Device: {config.DEVICE}")
    
    # Define loss function with weights for URLLC priority
    criterion = nn.MSELoss(reduction='none')
    
    def weighted_mse_loss(predictions, targets):
        """Custom weighted MSE loss"""
        losses = criterion(predictions, targets)  # (batch, 3)
        weighted_losses = losses * config.LOSS_WEIGHTS
        return weighted_losses.mean()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training tracking
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    
    print("\n" + "-" * 80)
    print("TRAINING CONFIGURATION")
    print("-" * 80)
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Optimizer: Adam")
    print(f"Loss function: Weighted MSE")
    print(f"  - eMBB weight: {config.LOSS_WEIGHTS[0].item():.1f}")
    print(f"  - URLLC weight: {config.LOSS_WEIGHTS[1].item():.1f} (prioritized)")
    print(f"  - mMTC weight: {config.LOSS_WEIGHTS[2].item():.1f}")
    print(f"LR scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
    print(f"Random seed: {config.RANDOM_SEED}")
    
    print("\n" + "-" * 80)
    print("DATASET INFORMATION")
    print("-" * 80)
    print(f"Train samples: {len(train_loader.dataset):,}")
    print(f"Val samples: {len(val_loader.dataset):,}")
    print(f"Test samples: {len(test_loader.dataset):,}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_one_epoch(model, train_loader, weighted_mse_loss, 
                                     optimizer, config.DEVICE, epoch, config.NUM_EPOCHS)
        
        # Validate
        val_loss = validate(model, val_loader, weighted_mse_loss, 
                           config.DEVICE, epoch, config.NUM_EPOCHS)
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Log losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\nEpoch [{epoch}/{config.NUM_EPOCHS}] Summary:")
        print(f"  train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
              f"lr: {new_lr:.6f} | time: {epoch_time:.2f}s")
        
        # LR reduction message
        if new_lr < old_lr:
            print(f"  >> Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
        
        # Save best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_path = os.path.join(config.MODEL_SAVE_DIR, config.MODEL_NAME)
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, 
                          best_model_path)
            print(f"  >> Best model saved! (val_loss: {val_loss:.4f})")
        
        # Evaluate with all metrics every 5 epochs
        if epoch % 5 == 0 or epoch == config.NUM_EPOCHS:
            print("\n" + "-" * 80)
            print(f"DETAILED EVALUATION - Epoch [{epoch}/{config.NUM_EPOCHS}]")
            print("-" * 80)
            
            # Evaluate on validation set
            val_metrics = evaluate_model(model, val_loader, config.DEVICE)
            print_metrics(val_metrics, title="Validation Set Metrics")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, 
                                          f"checkpoint_epoch_{epoch}.pth")
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, 
                          checkpoint_path)
        
        print("-" * 80)
    
    total_time = time.time() - start_time
    
    # Final summary
    print("\n" + "=" * 80)
    print(" " * 30 + "TRAINING COMPLETED")
    print("=" * 80)
    print(f"Total training time: {total_time/3600:.2f} hours ({total_time/60:.2f} minutes)")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")
    
    # Final evaluation on all splits
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON ALL SPLITS")
    print("=" * 80)
    
    # Load best model
    checkpoint = torch.load(os.path.join(config.MODEL_SAVE_DIR, config.MODEL_NAME))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on all splits
    print("\n>>> TRAINING SET:")
    train_metrics = evaluate_model(model, train_loader, config.DEVICE)
    print_metrics(train_metrics, title="Training Set - Final Metrics")
    
    print("\n>>> VALIDATION SET:")
    val_metrics = evaluate_model(model, val_loader, config.DEVICE)
    print_metrics(val_metrics, title="Validation Set - Final Metrics")
    
    print("\n>>> TEST SET:")
    test_metrics = evaluate_model(model, test_loader, config.DEVICE)
    print_metrics(test_metrics, title="Test Set - Final Metrics")
    
    print("\n" + "=" * 80)
    print(f"Best model saved at: {os.path.join(config.MODEL_SAVE_DIR, config.MODEL_NAME)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
