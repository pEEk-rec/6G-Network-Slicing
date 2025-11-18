"""
Evaluation metrics for 6G Bandwidth Allocation
Robust metrics without MAPE (removed due to slice-dependent instability)
"""

import torch
import numpy as np


def denormalize_predictions(predictions, targets, scaler):
    """
    Denormalize predictions and targets back to original scale
    Reverses: normalize → log → original Mbps values
    
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
    
    return torch.FloatTensor(pred_denorm), torch.FloatTensor(target_denorm)


def calculate_mae(predictions, targets):
    """
    Mean Absolute Error per slice (in Mbps)
    Lower is better. Interpretable: average error in Mbps
    """
    mae = torch.abs(predictions - targets).mean(dim=0)
    return {
        'mae_embb': mae[0].item(),
        'mae_urllc': mae[1].item(),
        'mae_mmtc': mae[2].item(),
        'mae_overall': mae.mean().item()
    }


def calculate_mse(predictions, targets):
    """
    Mean Squared Error per slice (Mbps²)
    Penalizes large errors more heavily than MAE
    """
    mse = ((predictions - targets) ** 2).mean(dim=0)
    return {
        'mse_embb': mse[0].item(),
        'mse_urllc': mse[1].item(),
        'mse_mmtc': mse[2].item(),
        'mse_overall': mse.mean().item()
    }


def calculate_rmse(predictions, targets):
    """
    Root Mean Squared Error per slice (in Mbps)
    Same units as MAE but penalizes outliers more
    """
    mse = ((predictions - targets) ** 2).mean(dim=0)
    rmse = torch.sqrt(mse)
    return {
        'rmse_embb': rmse[0].item(),
        'rmse_urllc': rmse[1].item(),
        'rmse_mmtc': rmse[2].item(),
        'rmse_overall': rmse.mean().item()
    }


def calculate_nrmse(predictions, targets):
    """
    Normalized RMSE (percentage of mean target value)
    Scale-independent metric: NRMSE = RMSE / mean(target) * 100
    
    Interpretation:
    - NRMSE < 10%: Excellent
    - NRMSE < 20%: Good
    - NRMSE < 30%: Acceptable
    - NRMSE > 30%: Poor
    """
    mse = ((predictions - targets) ** 2).mean(dim=0)
    rmse = torch.sqrt(mse)
    mean_targets = targets.mean(dim=0)
    nrmse = (rmse / (mean_targets + 1e-8)) * 100
    
    return {
        'nrmse_embb': nrmse[0].item(),
        'nrmse_urllc': nrmse[1].item(),
        'nrmse_mmtc': nrmse[2].item(),
        'nrmse_overall': nrmse.mean().item()
    }


def calculate_r2_score(predictions, targets):
    """
    R² Score (Coefficient of Determination)
    Measures how well predictions explain variance in targets
    
    Interpretation:
    - R² = 1.0: Perfect predictions
    - R² = 0.8-0.95: Excellent model
    - R² = 0.6-0.8: Good model
    - R² = 0.4-0.6: Moderate model
    - R² < 0.4: Poor model
    - R² < 0: Worse than predicting mean
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


def calculate_smape(predictions, targets):
    """
    Symmetric Mean Absolute Percentage Error
    More robust than MAPE for bandwidth allocation
    Range: 0-200% (lower is better)
    
    Interpretation:
    - SMAPE < 10%: Excellent
    - SMAPE < 20%: Good
    - SMAPE < 30%: Acceptable
    - SMAPE > 30%: Poor
    """
    numerator = torch.abs(predictions - targets)
    denominator = (torch.abs(predictions) + torch.abs(targets)) / 2
    smape = (numerator / (denominator + 1e-8) * 100).mean(dim=0)
    
    return {
        'smape_embb': smape[0].item(),
        'smape_urllc': smape[1].item(),
        'smape_mmtc': smape[2].item(),
        'smape_overall': smape.mean().item()
    }


def calculate_max_error(predictions, targets):
    """
    Maximum Absolute Error per slice
    Identifies worst-case prediction errors
    """
    max_err = torch.abs(predictions - targets).max(dim=0)[0]
    return {
        'max_error_embb': max_err[0].item(),
        'max_error_urllc': max_err[1].item(),
        'max_error_mmtc': max_err[2].item(),
        'max_error_overall': max_err.mean().item()
    }


def calculate_allocation_accuracy(predictions, targets, tolerance=0.1):
    """
    Allocation Accuracy: % of predictions within tolerance of target
    
    Args:
        tolerance: Acceptable error margin (default 10% = 0.1)
    
    Returns:
        Percentage of predictions within tolerance
    """
    relative_error = torch.abs((predictions - targets) / (targets + 1e-8))
    within_tolerance = (relative_error <= tolerance).float().mean(dim=0) * 100
    
    return {
        'accuracy_10pct_embb': within_tolerance[0].item(),
        'accuracy_10pct_urllc': within_tolerance[1].item(),
        'accuracy_10pct_mmtc': within_tolerance[2].item(),
        'accuracy_10pct_overall': within_tolerance.mean().item()
    }


def evaluate_model(model, dataloader, device, target_scaler=None):
    """
    Comprehensive evaluation with all metrics
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: torch.device (cuda/cpu)
        target_scaler: If provided, denormalize predictions for interpretable metrics
    
    Returns:
        dict: All evaluation metrics
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
    
    # Denormalize if scaler provided (for interpretable metrics in Mbps)
    if target_scaler is not None:
        all_predictions, all_targets = denormalize_predictions(
            all_predictions, all_targets, target_scaler
        )
    
    # Calculate all metrics
    metrics = {}
    metrics.update(calculate_mae(all_predictions, all_targets))
    metrics.update(calculate_rmse(all_predictions, all_targets))
    metrics.update(calculate_nrmse(all_predictions, all_targets))
    metrics.update(calculate_r2_score(all_predictions, all_targets))
    metrics.update(calculate_smape(all_predictions, all_targets))
    metrics.update(calculate_max_error(all_predictions, all_targets))
    metrics.update(calculate_allocation_accuracy(all_predictions, all_targets, tolerance=0.10))
    
    return metrics


def print_metrics(metrics, title="Evaluation Metrics"):
    """Pretty print metrics in organized table format"""
    print("\n" + "=" * 90)
    print(f"{title:^90}")
    print("=" * 90)
    
    # Primary metrics table
    print("\nPrimary Metrics (in Mbps):")
    print("-" * 90)
    print(f"{'Metric':<20} {'eMBB':>15} {'URLLC':>15} {'mMTC':>15} {'Overall':>15}")
    print("-" * 90)
    
    print(f"{'MAE':<20} {metrics['mae_embb']:>15.2f} {metrics['mae_urllc']:>15.2f} "
          f"{metrics['mae_mmtc']:>15.2f} {metrics['mae_overall']:>15.2f}")
    
    print(f"{'RMSE':<20} {metrics['rmse_embb']:>15.2f} {metrics['rmse_urllc']:>15.2f} "
          f"{metrics['rmse_mmtc']:>15.2f} {metrics['rmse_overall']:>15.2f}")
    
    print(f"{'Max Error':<20} {metrics['max_error_embb']:>15.2f} {metrics['max_error_urllc']:>15.2f} "
          f"{metrics['max_error_mmtc']:>15.2f} {metrics['max_error_overall']:>15.2f}")
    
    # Normalized/percentage metrics
    print("\nNormalized Metrics (%):")
    print("-" * 90)
    print(f"{'Metric':<20} {'eMBB':>15} {'URLLC':>15} {'mMTC':>15} {'Overall':>15}")
    print("-" * 90)
    
    print(f"{'NRMSE (%)':<20} {metrics['nrmse_embb']:>15.2f} {metrics['nrmse_urllc']:>15.2f} "
          f"{metrics['nrmse_mmtc']:>15.2f} {metrics['nrmse_overall']:>15.2f}")
    
    print(f"{'SMAPE (%)':<20} {metrics['smape_embb']:>15.2f} {metrics['smape_urllc']:>15.2f} "
          f"{metrics['smape_mmtc']:>15.2f} {metrics['smape_overall']:>15.2f}")
    
    print(f"{'Accuracy@10% (%)':<20} {metrics['accuracy_10pct_embb']:>15.2f} {metrics['accuracy_10pct_urllc']:>15.2f} "
          f"{metrics['accuracy_10pct_mmtc']:>15.2f} {metrics['accuracy_10pct_overall']:>15.2f}")
    
    # R² Score (most important)
    print("\nModel Performance (R² Score):")
    print("-" * 90)
    print(f"{'R² Score':<20} {metrics['r2_embb']:>15.4f} {metrics['r2_urllc']:>15.4f} "
          f"{metrics['r2_mmtc']:>15.4f} {metrics['r2_overall']:>15.4f}")
    
    # Interpretation guide
    print("\n" + "=" * 90)
    print("Interpretation Guide:")
    print("-" * 90)
    print("R² Score:        0.80-0.95 = Excellent | 0.60-0.80 = Good | <0.60 = Needs Improvement")
    print("NRMSE:           <15% = Excellent | <25% = Good | <35% = Acceptable")
    print("SMAPE:           <15% = Excellent | <25% = Good | <35% = Acceptable")
    print("Accuracy@10%:    >85% = Excellent | >70% = Good | >60% = Acceptable")
    print("=" * 90 + "\n")
