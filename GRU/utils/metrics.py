"""
Evaluation metrics for 6G Bandwidth Allocation - STREAMLINED VERSION
Focus: R² Score (model quality) and RMSE (prediction error in Mbps)
Removed: Clutter metrics that don't drive optimization decisions
"""

import torch
import numpy as np


def denormalize_predictions(predictions, targets, scaler):
    """
    Denormalize predictions and targets back to original scale
    Reverses: StandardScaler → log1p → original Mbps values
    
    Args:
        predictions: (N, 3) normalized tensor
        targets: (N, 3) normalized tensor
        scaler: StandardScaler used for normalization
    
    Returns:
        tuple: (denorm_predictions, denorm_targets) in original Mbps scale
    """
    pred_numpy = predictions.cpu().numpy()
    target_numpy = targets.cpu().numpy()
    
    # Step 1: Inverse StandardScaler
    pred_denorm = scaler.inverse_transform(pred_numpy)
    target_denorm = scaler.inverse_transform(target_numpy)
    
    # Step 2: Inverse log transform (expm1 reverses log1p)
    pred_denorm = np.expm1(pred_denorm)
    target_denorm = np.expm1(target_denorm)
    
    # Clip negative values (shouldn't happen but safety check)
    pred_denorm = np.maximum(pred_denorm, 0)
    target_denorm = np.maximum(target_denorm, 0)
    
    return torch.FloatTensor(pred_denorm), torch.FloatTensor(target_denorm)


def calculate_rmse(predictions, targets):
    """
    Root Mean Squared Error per slice (in Mbps)
    PRIMARY METRIC: Measures average prediction error magnitude
    
    Lower is better. Units: Mbps (interpretable)
    """
    mse = ((predictions - targets) ** 2).mean(dim=0)
    rmse = torch.sqrt(mse)
    
    return {
        'rmse_embb': rmse[0].item(),
        'rmse_urllc': rmse[1].item(),
        'rmse_mmtc': rmse[2].item(),
        'rmse_overall': rmse.mean().item()
    }


def calculate_r2_score(predictions, targets):
    """
    R² Score (Coefficient of Determination)
    PRIMARY METRIC: Measures model's explanatory power
    
    Interpretation:
    - R² = 1.0: Perfect predictions
    - R² = 0.8-1.0: Excellent model
    - R² = 0.6-0.8: Good model
    - R² = 0.4-0.6: Moderate model
    - R² < 0.4: Poor model
    - R² < 0: Worse than predicting mean (FAILURE)
    """
    ss_res = ((targets - predictions) ** 2).sum(dim=0)
    ss_tot = ((targets - targets.mean(dim=0)) ** 2).sum(dim=0)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'r2_embb': r2[0].item(),
        'r2_urllc': r2[1].item(),
        'r2_mmtc': r2[2].item(),
        'r2_overall': r2.mean().item()
    }


def calculate_mae(predictions, targets):
    """
    Mean Absolute Error per slice (in Mbps)
    AUXILIARY METRIC: L1 distance, less sensitive to outliers than RMSE
    """
    mae = torch.abs(predictions - targets).mean(dim=0)
    return {
        'mae_embb': mae[0].item(),
        'mae_urllc': mae[1].item(),
        'mae_mmtc': mae[2].item(),
        'mae_overall': mae.mean().item()
    }


def get_per_slice_loss(predictions, targets, loss_fn):
    """
    Calculate loss for each slice independently
    CRITICAL for debugging: Shows which slices are failing
    
    Args:
        predictions: (N, 3) tensor
        targets: (N, 3) tensor
        loss_fn: torch.nn loss function (MSELoss or L1Loss)
    
    Returns:
        dict: Individual slice losses
    """
    losses = {}
    for i, slice_name in enumerate(['embb', 'urllc', 'mmtc']):
        slice_loss = loss_fn(predictions[:, i], targets[:, i])
        losses[f'loss_{slice_name}'] = slice_loss.item()
    
    # Overall loss (mean of slices)
    losses['loss_overall'] = sum(losses.values()) / 3
    
    return losses


def evaluate_model(model, dataloader, device, target_scaler=None, loss_fn=None):
    """
    Comprehensive evaluation with ESSENTIAL metrics only
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: torch.device (cuda/cpu)
        target_scaler: If provided, denormalize for interpretable metrics
        loss_fn: Loss function to compute per-slice losses (optional)
    
    Returns:
        dict: Essential evaluation metrics (RMSE, R², MAE, per-slice losses)
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            predictions = model(X_batch)
            all_predictions.append(predictions)
            all_targets.append(y_batch)
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate per-slice loss in NORMALIZED space (before denormalization)
    metrics = {}
    if loss_fn is not None:
        metrics.update(get_per_slice_loss(all_predictions, all_targets, loss_fn))
    
    # Denormalize if scaler provided (for interpretable metrics in Mbps)
    if target_scaler is not None:
        all_predictions, all_targets = denormalize_predictions(
            all_predictions, all_targets, target_scaler
        )
    
    # Calculate essential metrics in Mbps space
    metrics.update(calculate_rmse(all_predictions, all_targets))
    metrics.update(calculate_r2_score(all_predictions, all_targets))
    metrics.update(calculate_mae(all_predictions, all_targets))
    
    return metrics


def print_metrics(metrics, title="Evaluation Metrics", show_loss=False):
    """
    Clean, focused metric printing
    Focus: R² (model quality) and RMSE (error magnitude)
    """
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    
    # Per-slice loss (if available) - Shows optimization behavior
    if show_loss and 'loss_embb' in metrics:
        print("\nPer-Slice Loss (Normalized Space):")
        print("-" * 80)
        print(f"  eMBB:    {metrics['loss_embb']:.4f}")
        print(f"  URLLC:   {metrics['loss_urllc']:.4f}")
        print(f"  mMTC:    {metrics['loss_mmtc']:.4f}")
        print(f"  Overall: {metrics['loss_overall']:.4f}")
    
    # Primary metrics: R² and RMSE
    print("\nPrimary Metrics:")
    print("-" * 80)
    print(f"{'Metric':<15} {'eMBB':>12} {'URLLC':>12} {'mMTC':>12} {'Overall':>12}")
    print("-" * 80)
    
    print(f"{'R² Score':<15} "
          f"{metrics['r2_embb']:>12.4f} "
          f"{metrics['r2_urllc']:>12.4f} "
          f"{metrics['r2_mmtc']:>12.4f} "
          f"{metrics['r2_overall']:>12.4f}")
    
    print(f"{'RMSE (Mbps)':<15} "
          f"{metrics['rmse_embb']:>12.2f} "
          f"{metrics['rmse_urllc']:>12.2f} "
          f"{metrics['rmse_mmtc']:>12.2f} "
          f"{metrics['rmse_overall']:>12.2f}")
    
    print(f"{'MAE (Mbps)':<15} "
          f"{metrics['mae_embb']:>12.2f} "
          f"{metrics['mae_urllc']:>12.2f} "
          f"{metrics['mae_mmtc']:>12.2f} "
          f"{metrics['mae_overall']:>12.2f}")
    
    # Interpretation
    print("\n" + "=" * 80)
    print("Interpretation:")
    print("-" * 80)
    
    r2_overall = metrics['r2_overall']
    rmse_overall = metrics['rmse_overall']
    
    # R² interpretation
    if r2_overall >= 0.8:
        r2_status = "✓ EXCELLENT"
    elif r2_overall >= 0.6:
        r2_status = "✓ GOOD"
    elif r2_overall >= 0.4:
        r2_status = "⚠ MODERATE"
    elif r2_overall >= 0.0:
        r2_status = "✗ POOR"
    else:
        r2_status = "✗✗ FAILURE (worse than mean baseline)"
    
    print(f"  R² Score:  {r2_overall:.4f} → {r2_status}")
    print(f"  RMSE:      {rmse_overall:.2f} Mbps (lower is better)")
    
    # Flag problematic slices
    problem_slices = []
    if metrics['r2_embb'] < 0.4:
        problem_slices.append('eMBB')
    if metrics['r2_urllc'] < 0.4:
        problem_slices.append('URLLC')
    if metrics['r2_mmtc'] < 0.4:
        problem_slices.append('mMTC')
    
    if problem_slices:
        print(f"\n⚠ WARNING: Poor R² for slices: {', '.join(problem_slices)}")
    
    print("=" * 80 + "\n")


def print_training_metrics(epoch, train_loss, val_loss, train_metrics, val_metrics, lr):
    """
    Compact training progress display
    Shows: Loss, R², RMSE for quick monitoring
    """
    print(f"\nEpoch [{epoch}] Summary:")
    print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.6f}")
    print(f"  Train R²: {train_metrics['r2_overall']:.4f} | Val R²: {val_metrics['r2_overall']:.4f}")
    print(f"  Train RMSE: {train_metrics['rmse_overall']:.2f} | Val RMSE: {val_metrics['rmse_overall']:.2f}")
    
    # Flag if validation is getting worse
    if val_metrics['r2_overall'] < train_metrics['r2_overall'] - 0.1:
        print("  ⚠ Warning: Overfitting detected (Val R² << Train R²)")