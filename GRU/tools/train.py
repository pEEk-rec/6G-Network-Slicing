"""
Training script for 6G Bandwidth Allocation GRU Model
FIXED VERSION - Proper loss tracking and model integration
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from tqdm import tqdm

from config import config
from data.dataloader import get_dataloaders
from models.gru_model import BandwidthAllocationGRU
from utils.helpers import save_checkpoint, count_parameters
from utils.metrics import evaluate_model, print_metrics, print_training_metrics


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


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs, track_per_slice=False):
    """
    Train model for one epoch
    
    Returns:
        float: Average training loss
        dict: Per-slice losses (if tracking enabled)
    """
    model.train()
    loss_meter = AverageMeter()
    
    # Per-slice loss tracking
    if track_per_slice:
        slice_loss_meters = {
            'embb': AverageMeter(),
            'urllc': AverageMeter(),
            'mmtc': AverageMeter()
        }
    
    pbar = tqdm(train_loader, desc=f'Epoch [{epoch}/{num_epochs}] Train', 
                ncols=120, ascii=True)
    
    for batch_idx, (X_batch, y_batch) in enumerate(pbar):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        # Track per-slice losses before backward pass
        if track_per_slice:
            with torch.no_grad():
                slice_losses = ((predictions - y_batch) ** 2).mean(dim=0)
                slice_loss_meters['embb'].update(slice_losses[0].item(), X_batch.size(0))
                slice_loss_meters['urllc'].update(slice_losses[1].item(), X_batch.size(0))
                slice_loss_meters['mmtc'].update(slice_losses[2].item(), X_batch.size(0))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP_NORM)
        optimizer.step()
        
        loss_meter.update(loss.item(), X_batch.size(0))
        
        pbar.set_postfix({
            'loss': f'{loss_meter.val:.4f}',
            'avg_loss': f'{loss_meter.avg:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    if track_per_slice:
        per_slice_losses = {
            'embb': slice_loss_meters['embb'].avg,
            'urllc': slice_loss_meters['urllc'].avg,
            'mmtc': slice_loss_meters['mmtc'].avg
        }
        return loss_meter.avg, per_slice_losses
    else:
        return loss_meter.avg, None


def validate(model, val_loader, criterion, device, epoch, num_epochs):
    """
    Validate model
    
    Returns:
        float: Average validation loss
    """
    model.eval()
    loss_meter = AverageMeter()
    
    pbar = tqdm(val_loader, desc=f'Epoch [{epoch}/{num_epochs}] Val  ', 
                ncols=120, ascii=True)
    
    with torch.no_grad():
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            loss_meter.update(loss.item(), X_batch.size(0))
            
            pbar.set_postfix({
                'loss': f'{loss_meter.val:.4f}',
                'avg_loss': f'{loss_meter.avg:.4f}'
            })
    
    return loss_meter.avg


def main():
    """
    Main training function
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "6G BANDWIDTH ALLOCATION GRU TRAINING")
    print("=" * 80)
    
    # Set random seed
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create dataloaders
    train_loader, val_loader, test_loader, target_scaler = get_dataloaders(config)
    
    # Initialize model
    print("\n" + "-" * 80)
    print("MODEL INITIALIZATION")
    print("-" * 80)
    
    # Check if 3-layer model is configured
    has_third_layer = hasattr(config, 'HIDDEN_SIZE_3')
    
    if has_third_layer:
        model = BandwidthAllocationGRU(
            input_size=config.INPUT_SIZE,
            hidden_size_1=config.HIDDEN_SIZE_1,
            hidden_size_2=config.HIDDEN_SIZE_2,
            hidden_size_3=config.HIDDEN_SIZE_3,
            dense_size=config.DENSE_SIZE,
            output_size=config.OUTPUT_SIZE,
            dropout=config.DROPOUT
        )
    else:
        model = BandwidthAllocationGRU(
            input_size=config.INPUT_SIZE,
            hidden_size_1=config.HIDDEN_SIZE_1,
            hidden_size_2=config.HIDDEN_SIZE_2,
            dense_size=config.DENSE_SIZE,
            output_size=config.OUTPUT_SIZE,
            dropout=config.DROPOUT
        )
    
    model = model.to(config.DEVICE)
    
    num_params = count_parameters(model)
    print(f"Model: BandwidthAllocationGRU")
    print(f"Total params: {num_params:,}")
    print(f"Device: {config.DEVICE}")
    
    # Define loss function
    if hasattr(config, 'USE_MAE_LOSS') and config.USE_MAE_LOSS:
        criterion_base = nn.L1Loss(reduction='none')
        loss_name = "MAE"
    else:
        criterion_base = nn.MSELoss(reduction='none')
        loss_name = "MSE"
    
    def weighted_loss(predictions, targets):
        """Weighted loss with equal or custom weights"""
        losses = criterion_base(predictions, targets)
        weighted_losses = losses * config.LOSS_WEIGHTS
        return weighted_losses.mean()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # LR Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config.LR_SCHEDULER_FACTOR, 
        patience=config.LR_SCHEDULER_PATIENCE,
        min_lr=config.MIN_LR
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA
    )
    
    # Training tracking
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    learning_rates = []
    
    # Check if per-slice tracking is enabled
    track_per_slice = hasattr(config, 'TRACK_PER_SLICE_LOSS') and config.TRACK_PER_SLICE_LOSS
    
    print("\n" + "-" * 80)
    print("TRAINING CONFIGURATION")
    print("-" * 80)
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Optimizer: AdamW (weight_decay={config.WEIGHT_DECAY})")
    print(f"Loss function: Weighted {loss_name}")
    print(f"  Weights: {config.LOSS_WEIGHTS.cpu().numpy()}")
    print(f"Gradient clipping: {config.GRADIENT_CLIP_NORM}")
    print(f"Early stopping: patience={config.EARLY_STOPPING_PATIENCE}")
    print(f"Per-slice tracking: {track_per_slice}")
    
    print("\n" + "-" * 80)
    print("DATASET INFORMATION")
    print("-" * 80)
    print(f"Train: {len(train_loader.dataset):,} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_loader.dataset):,} samples, {len(val_loader)} batches")
    print(f"Test:  {len(test_loader.dataset):,} samples, {len(test_loader)} batches")
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        
        # Train
        train_loss, per_slice_losses = train_one_epoch(
            model, train_loader, weighted_loss, optimizer, 
            config.DEVICE, epoch, config.NUM_EPOCHS, track_per_slice
        )
        
        # Validate
        val_loss = validate(model, val_loader, weighted_loss, 
                           config.DEVICE, epoch, config.NUM_EPOCHS)
        
        # Update LR
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Log
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        learning_rates.append(new_lr)
        
        epoch_time = time.time() - epoch_start_time
        
        # Print summary
        print(f"\nEpoch [{epoch}/{config.NUM_EPOCHS}] Summary:")
        print(f"  train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
              f"lr: {new_lr:.6f} | time: {epoch_time:.2f}s")
        
        # Per-slice loss tracking
        if track_per_slice and per_slice_losses:
            print(f"  Per-slice train loss: eMBB={per_slice_losses['embb']:.4f}, "
                  f"URLLC={per_slice_losses['urllc']:.4f}, mMTC={per_slice_losses['mmtc']:.4f}")
        
        if new_lr < old_lr:
            print(f"  >> Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_path = os.path.join(config.MODEL_SAVE_DIR, config.MODEL_NAME)
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, best_model_path)
            print(f"  >> Best model saved! (val_loss: {val_loss:.4f})")
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"\n  >> Early stopping triggered after {epoch} epochs")
            print(f"  >> Best epoch was {best_epoch} with val_loss: {best_val_loss:.4f}")
            break
        
        # Detailed evaluation
        if epoch % config.LOG_INTERVAL == 0 or epoch == config.NUM_EPOCHS:
            print("\n" + "-" * 80)
            print(f"DETAILED EVALUATION - Epoch [{epoch}/{config.NUM_EPOCHS}]")
            print("-" * 80)
            
            val_metrics = evaluate_model(model, val_loader, config.DEVICE, target_scaler, criterion_base)
            print_metrics(val_metrics, title="Validation Set Metrics", show_loss=track_per_slice)
        
        print("-" * 80)
    
    total_time = time.time() - start_time
    
    # Save final model
    final_checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, config.FINAL_MODEL_NAME)
    save_checkpoint(model, optimizer, epoch, train_losses[-1], val_losses[-1], final_checkpoint_path)
    
    # Save history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss
    }
    history_path = os.path.join(config.MODEL_SAVE_DIR, "training_history.npy")
    np.save(history_path, history)
    
    # Final summary
    print("\n" + "=" * 80)
    print(" " * 30 + "TRAINING COMPLETED")
    print("=" * 80)
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.2f} minutes)")
    print(f"Total epochs: {epoch}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.4f}")
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON ALL SPLITS")
    print("=" * 80)
    
    checkpoint = torch.load(os.path.join(config.MODEL_SAVE_DIR, config.MODEL_NAME))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\n>>> TRAINING SET:")
    train_metrics = evaluate_model(model, train_loader, config.DEVICE, target_scaler, criterion_base)
    print_metrics(train_metrics, title="Training Set - Final Metrics", show_loss=True)
    
    print("\n>>> VALIDATION SET:")
    val_metrics = evaluate_model(model, val_loader, config.DEVICE, target_scaler, criterion_base)
    print_metrics(val_metrics, title="Validation Set - Final Metrics", show_loss=True)
    
    print("\n>>> TEST SET:")
    test_metrics = evaluate_model(model, test_loader, config.DEVICE, target_scaler, criterion_base)
    print_metrics(test_metrics, title="Test Set - Final Metrics", show_loss=True)
    
    print("\n" + "=" * 80)
    print("SAVED MODELS:")
    print("-" * 80)
    print(f"Best model:  {os.path.join(config.MODEL_SAVE_DIR, config.MODEL_NAME)}")
    print(f"Final model: {final_checkpoint_path}")
    print(f"History:     {history_path}")
    
    print("\n" + "=" * 80)
    print("KEY METRICS (Test Set)")
    print("=" * 80)
    print(f"R² Score:    {test_metrics['r2_overall']:.4f}")
    print(f"RMSE (Mbps): {test_metrics['rmse_overall']:.2f}")
    print(f"MAE (Mbps):  {test_metrics['mae_overall']:.2f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
