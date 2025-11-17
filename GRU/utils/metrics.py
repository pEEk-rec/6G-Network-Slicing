"""
Evaluation metrics for 6G Bandwidth Allocation
"""

import torch
import numpy as np


def calculate_mae(predictions, targets):
    """Mean Absolute Error per slice"""
    mae = torch.abs(predictions - targets).mean(dim=0)
    return {
        'mae_embb': mae[0].item(),
        'mae_urllc': mae[1].item(),
        'mae_mmtc': mae[2].item(),
        'mae_overall': mae.mean().item()
    }


def calculate_mse(predictions, targets):
    """Mean Squared Error per slice"""
    mse = ((predictions - targets) ** 2).mean(dim=0)
    return {
        'mse_embb': mse[0].item(),
        'mse_urllc': mse[1].item(),
        'mse_mmtc': mse[2].item(),
        'mse_overall': mse.mean().item()
    }


def calculate_rmse(predictions, targets):
    """Root Mean Squared Error per slice"""
    mse = ((predictions - targets) ** 2).mean(dim=0)
    rmse = torch.sqrt(mse)
    return {
        'rmse_embb': rmse[0].item(),
        'rmse_urllc': rmse[1].item(),
        'rmse_mmtc': rmse[2].item(),
        'rmse_overall': rmse.mean().item()
    }


def calculate_mape(predictions, targets, epsilon=1e-8):
    """
    Mean Absolute Percentage Error (MAPE)
    Perfect for bandwidth allocation - shows % error relative to actual allocation
    """
    mape = (torch.abs((targets - predictions) / (targets + epsilon)) * 100).mean(dim=0)
    return {
        'mape_embb': mape[0].item(),
        'mape_urllc': mape[1].item(),
        'mape_mmtc': mape[2].item(),
        'mape_overall': mape.mean().item()
    }


def calculate_r2_score(predictions, targets):
    """R² Score (Coefficient of Determination)"""
    ss_res = ((targets - predictions) ** 2).sum(dim=0)
    ss_tot = ((targets - targets.mean(dim=0)) ** 2).sum(dim=0)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    return {
        'r2_embb': r2[0].item(),
        'r2_urllc': r2[1].item(),
        'r2_mmtc': r2[2].item(),
        'r2_overall': r2.mean().item()
    }


def evaluate_model(model, dataloader, device):
    """
    Comprehensive evaluation with all metrics
    
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
    
    # Calculate all metrics
    metrics = {}
    metrics.update(calculate_mae(all_predictions, all_targets))
    metrics.update(calculate_mse(all_predictions, all_targets))
    metrics.update(calculate_rmse(all_predictions, all_targets))
    metrics.update(calculate_mape(all_predictions, all_targets))
    metrics.update(calculate_r2_score(all_predictions, all_targets))
    
    return metrics


def print_metrics(metrics, title="Evaluation Metrics"):
    """Pretty print metrics"""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    
    # Per-slice metrics
    print("\nPer-Slice Metrics:")
    print("-" * 80)
    print(f"{'Metric':<15} {'eMBB':>15} {'URLLC':>15} {'mMTC':>15} {'Overall':>15}")
    print("-" * 80)
    
    print(f"{'MAE':<15} {metrics['mae_embb']:>15.4f} {metrics['mae_urllc']:>15.4f} "
          f"{metrics['mae_mmtc']:>15.4f} {metrics['mae_overall']:>15.4f}")
    
    print(f"{'RMSE':<15} {metrics['rmse_embb']:>15.4f} {metrics['rmse_urllc']:>15.4f} "
          f"{metrics['rmse_mmtc']:>15.4f} {metrics['rmse_overall']:>15.4f}")
    
    print(f"{'MAPE (%)':<15} {metrics['mape_embb']:>15.2f} {metrics['mape_urllc']:>15.2f} "
          f"{metrics['mape_mmtc']:>15.2f} {metrics['mape_overall']:>15.2f}")
    
    print(f"{'R² Score':<15} {metrics['r2_embb']:>15.4f} {metrics['r2_urllc']:>15.4f} "
          f"{metrics['r2_mmtc']:>15.4f} {metrics['r2_overall']:>15.4f}")
    
    print("=" * 80 + "\n")
