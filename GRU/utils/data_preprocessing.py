"""
Preprocess raw NS-3 dataset for GRU training
Converts per-UE data to per-timestep per-slice aggregated format
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os


def aggregate_per_timestep(df):
    """
    Aggregate per-UE data into per-timestep format
    
    Input: DataFrame with 3 rows per timestep (one per slice_type)
    Output: DataFrame with 1 row per timestep (all slices aggregated)
    """
    
    print("\n" + "=" * 80)
    print("AGGREGATING PER-UE DATA TO PER-TIMESTEP FORMAT")
    print("=" * 80)
    print(f"Input: {len(df)} rows (per-UE)")
    
    # Group by (episode_id, step_id) to aggregate each timestep
    aggregated_rows = []
    
    for (episode_id, step_id), group in df.groupby(['episode_id', 'step_id']):
        # Should have 3 rows (one per slice_type: 1, 2, 3)
        assert len(group) == 3, f"Expected 3 rows per timestep, got {len(group)}"
        
        # Extract per-slice data
        embb_row = group[group['slice_type'] == 1].iloc[0]
        urllc_row = group[group['slice_type'] == 2].iloc[0]
        mmtc_row = group[group['slice_type'] == 3].iloc[0]
        
        # Create aggregated row
        agg_row = {
            # === Metadata ===
            'episode_id': episode_id,
            'step_id': step_id,
            
            # === Temporal Context (3 features) ===
            'timestamp': embb_row['timestamp'],
            'time_hour': embb_row['time_hour'],
            'is_peak_hour': embb_row['is_peak_hour'],
            
            # === Network Load (2 features) ===
            'network_utilization': embb_row['network_utilization'],
            'total_active_users': embb_row['total_active_users'],
            
            # === Per-Slice Demands (3 features) ===
            'total_demand_embb': embb_row['total_demand'],
            'total_demand_urllc': urllc_row['total_demand'],
            'total_demand_mmtc': mmtc_row['total_demand'],
            
            # === Per-Slice SINR (3 features) ===
            'avg_sinr_embb': embb_row['sinr_db'],
            'avg_sinr_urllc': urllc_row['sinr_db'],
            'avg_sinr_mmtc': mmtc_row['sinr_db'],
            
            # === Per-Slice CQI (3 features) ===
            'avg_cqi_embb': embb_row['cqi'],
            'avg_cqi_urllc': urllc_row['cqi'],
            'avg_cqi_mmtc': mmtc_row['cqi'],
            
            # === Slice Utilizations (3 features) ===
            'embb_utilization': embb_row['embb_utilization'],
            'urllc_utilization': embb_row['urllc_utilization'],
            'mmtc_utilization': embb_row['mmtc_utilization'],
            
            # === Target Variables (3 outputs) ===
            'allocated_bandwidth_mbps_embb': embb_row['allocated_bandwidth_mbps'],
            'allocated_bandwidth_mbps_urllc': urllc_row['allocated_bandwidth_mbps'],
            'allocated_bandwidth_mbps_mmtc': mmtc_row['allocated_bandwidth_mbps'],
        }
        
        aggregated_rows.append(agg_row)
    
    df_agg = pd.DataFrame(aggregated_rows)
    print(f"Output: {len(df_agg)} rows (per-timestep)")
    print(f"Reduction: {len(df)//len(df_agg)}:1")
    
    return df_agg


def split_episodes(df, train_episodes, val_episodes, test_episodes):
    """
    Split data by episode IDs (prevent data leakage)
    """
    print("\n" + "=" * 80)
    print("SPLITTING BY EPISODES")
    print("=" * 80)
    
    train_df = df[df['episode_id'].isin(train_episodes)].reset_index(drop=True)
    val_df = df[df['episode_id'].isin(val_episodes)].reset_index(drop=True)
    test_df = df[df['episode_id'].isin(test_episodes)].reset_index(drop=True)
    
    print(f"Train: Episodes {min(train_episodes)}-{max(train_episodes)} → {len(train_df)} timesteps")
    print(f"Val:   Episodes {min(val_episodes)}-{max(val_episodes)} → {len(val_df)} timesteps")
    print(f"Test:  Episodes {min(test_episodes)}-{max(test_episodes)} → {len(test_df)} timesteps")
    
    return train_df, val_df, test_df


def normalize_features(train_df, val_df, test_df, save_dir):
    """
    Normalize features using StandardScaler fitted on training data only
    """
    print("\n" + "=" * 80)
    print("FEATURE NORMALIZATION (StandardScaler)")
    print("=" * 80)
    
    # Define feature and target columns
    feature_cols = [
        'timestamp', 'time_hour', 'is_peak_hour',
        'network_utilization', 'total_active_users',
        'total_demand_embb', 'total_demand_urllc', 'total_demand_mmtc',
        'avg_sinr_embb', 'avg_sinr_urllc', 'avg_sinr_mmtc',
        'avg_cqi_embb', 'avg_cqi_urllc', 'avg_cqi_mmtc',
        'embb_utilization', 'urllc_utilization', 'mmtc_utilization'
    ]
    
    target_cols = [
        'allocated_bandwidth_mbps_embb',
        'allocated_bandwidth_mbps_urllc',
        'allocated_bandwidth_mbps_mmtc'
    ]
    
    # Fit scaler on TRAINING data only
    X_scaler = StandardScaler()
    X_scaler.fit(train_df[feature_cols])
    
    print("Feature Scaler Statistics (from training data):")
    print(f"  Mean: {X_scaler.mean_[:5]}... (first 5 features)")
    print(f"  Std:  {X_scaler.scale_[:5]}... (first 5 features)")
    
    # Transform all splits using TRAINING statistics
    train_df[feature_cols] = X_scaler.transform(train_df[feature_cols])
    val_df[feature_cols] = X_scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = X_scaler.transform(test_df[feature_cols])
    
    # Normalize targets (log-scale + StandardScaler for wide range)
    print("\n" + "=" * 80)
    print("TARGET NORMALIZATION (Log + StandardScaler)")
    print("=" * 80)
    
    # Apply log1p transform (handles zeros/small values)
    train_targets_log = np.log1p(train_df[target_cols].values)
    val_targets_log = np.log1p(val_df[target_cols].values)
    test_targets_log = np.log1p(test_df[target_cols].values)
    
    print("Target ranges BEFORE log transform:")
    print(f"  Min:  {train_df[target_cols].min().values}")
    print(f"  Max:  {train_df[target_cols].max().values}")
    print(f"  Mean: {train_df[target_cols].mean().values}")
    
    # Fit target scaler on training data
    y_scaler = StandardScaler()
    y_scaler.fit(train_targets_log)
    
    # Transform all splits
    train_df[target_cols] = y_scaler.transform(train_targets_log)
    val_df[target_cols] = y_scaler.transform(val_targets_log)
    test_df[target_cols] = y_scaler.transform(test_targets_log)
    
    print("\nTarget ranges AFTER log + normalization:")
    print(f"  Min:  {train_df[target_cols].min().values}")
    print(f"  Max:  {train_df[target_cols].max().values}")
    print(f"  Mean: {train_df[target_cols].mean().values}")
    
    # Save scalers
    joblib.dump(X_scaler, os.path.join(save_dir, 'feature_scaler.pkl'))
    joblib.dump(y_scaler, os.path.join(save_dir, 'target_scaler.pkl'))
    print(f"\n✓ Scalers saved to {save_dir}")
    
    return train_df, val_df, test_df, X_scaler, y_scaler


def main():
    """
    Main preprocessing pipeline
    """
    print("\n" + "=" * 80)
    print("6G BANDWIDTH ALLOCATION - DATA PREPROCESSING")
    print("=" * 80)
    
    # Paths
    raw_data_path = "D:/Academics/SEM-5/Machine Learning/ML_courseproj/Dataset/nr_dataset_full.csv"
    save_dir = "D:/Academics/SEM-5/Machine Learning/ML_courseproj/Processed"
    os.makedirs(save_dir, exist_ok=True)
    
    # Load raw data
    print(f"\nLoading data from: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Aggregate per-UE to per-timestep
    df_agg = aggregate_per_timestep(df)
    
    # Split by episodes (0-79 train, 80-89 val, 90-99 test)
    train_episodes = list(range(0, 80))
    val_episodes = list(range(80, 90))
    test_episodes = list(range(90, 100))
    
    train_df, val_df, test_df = split_episodes(df_agg, train_episodes, val_episodes, test_episodes)
    
    # Normalize features and targets
    train_df, val_df, test_df, X_scaler, y_scaler = normalize_features(
        train_df, val_df, test_df, save_dir
    )
    
    # Save processed data
    print("\n" + "=" * 80)
    print("SAVING PROCESSED DATA")
    print("=" * 80)
    
    train_df.to_csv(os.path.join(save_dir, 'train_processed.csv'), index=False)
    val_df.to_csv(os.path.join(save_dir, 'val_processed.csv'), index=False)
    test_df.to_csv(os.path.join(save_dir, 'test_processed.csv'), index=False)
    
    print(f"✓ Saved train_processed.csv ({len(train_df)} rows)")
    print(f"✓ Saved val_processed.csv ({len(val_df)} rows)")
    print(f"✓ Saved test_processed.csv ({len(test_df)} rows)")
    print("=" * 80)


if __name__ == "__main__":
    main()
