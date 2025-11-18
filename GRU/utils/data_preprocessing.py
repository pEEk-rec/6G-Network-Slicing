"""
FIXED Preprocessing Pipeline for 6G Bandwidth Allocation
Major improvements:
- Removed timestamp (non-predictive)
- Added feature engineering (temporal, ratios, interactions)
- Better normalization (RobustScaler for features, targets handled separately per slice)
- Data quality verification
- Proper handling of scale imbalance
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import os


def aggregate_per_timestep(df):
    """
    Aggregate per-UE data into per-timestep format
    """
    print("\n" + "=" * 80)
    print("AGGREGATING PER-UE DATA TO PER-TIMESTEP FORMAT")
    print("=" * 80)
    print(f"Input: {len(df)} rows (per-UE)")
    
    aggregated_rows = []
    
    for (episode_id, step_id), group in df.groupby(['episode_id', 'step_id']):
        assert len(group) == 3, f"Expected 3 rows per timestep, got {len(group)}"
        
        embb_row = group[group['slice_type'] == 1].iloc[0]
        urllc_row = group[group['slice_type'] == 2].iloc[0]
        mmtc_row = group[group['slice_type'] == 3].iloc[0]
        
        agg_row = {
            # === Metadata ===
            'episode_id': episode_id,
            'step_id': step_id,
            
            # === Temporal Context (2 features) - REMOVED timestamp ===
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
    
    return df_agg


def engineer_features(df):
    """
    Add engineered features for better predictive power
    """
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)
    
    # Sort by episode and step for temporal operations
    df = df.sort_values(['episode_id', 'step_id']).reset_index(drop=True)
    
    # === 1. Demand Ratios (relative importance of each slice) ===
    total_demand = (df['total_demand_embb'] + 
                    df['total_demand_urllc'] + 
                    df['total_demand_mmtc'] + 1e-8)
    
    df['embb_demand_ratio'] = df['total_demand_embb'] / total_demand
    df['urllc_demand_ratio'] = df['total_demand_urllc'] / total_demand
    df['mmtc_demand_ratio'] = df['total_demand_mmtc'] / total_demand
    
    print("✓ Added demand ratios (3 features)")
    
    # === 2. Load per User (network efficiency) ===
    df['load_per_user'] = df['network_utilization'] / (df['total_active_users'] + 1)
    
    print("✓ Added load per user (1 feature)")
    
    # === 3. SINR-CQI Interaction (channel quality indicator) ===
    df['embb_channel_quality'] = df['avg_sinr_embb'] * df['avg_cqi_embb']
    df['urllc_channel_quality'] = df['avg_sinr_urllc'] * df['avg_cqi_urllc']
    df['mmtc_channel_quality'] = df['avg_sinr_mmtc'] * df['avg_cqi_mmtc']
    
    print("✓ Added channel quality features (3 features)")
    
    # === 4. Temporal Features (changes over time) ===
    # Calculate per-episode to avoid leakage across episodes
    for col in ['total_demand_embb', 'total_demand_urllc', 'total_demand_mmtc',
                'network_utilization', 'total_active_users']:
        # Delta (change from previous timestep)
        df[f'{col}_delta'] = df.groupby('episode_id')[col].diff().fillna(0)
        
        # Moving average (last 3 timesteps)
        df[f'{col}_ma3'] = (df.groupby('episode_id')[col]
                            .rolling(window=3, min_periods=1)
                            .mean()
                            .reset_index(0, drop=True))
    
    print("✓ Added temporal features: delta + MA3 (10 features)")
    
    # === 5. Cyclic encoding for time_hour (better than raw hour) ===
    df['time_hour_sin'] = np.sin(2 * np.pi * df['time_hour'] / 24)
    df['time_hour_cos'] = np.cos(2 * np.pi * df['time_hour'] / 24)
    
    print("✓ Added cyclic time encoding (2 features)")
    
    # === 6. Utilization-Demand Mismatch (are we under/over-allocating?) ===
    df['embb_util_demand_ratio'] = df['embb_utilization'] / (df['total_demand_embb'] + 1e-8)
    df['urllc_util_demand_ratio'] = df['urllc_utilization'] / (df['total_demand_urllc'] + 1e-8)
    df['mmtc_util_demand_ratio'] = df['mmtc_utilization'] / (df['total_demand_mmtc'] + 1e-8)
    
    print("✓ Added utilization-demand ratios (3 features)")
    
    print(f"\nTotal features: {len(get_feature_columns())} (was 17, now enriched)")
    
    return df


def get_feature_columns():
    """
    Define ALL feature columns (base + engineered)
    """
    base_features = [
        # Temporal (3 features - removed timestamp, added cyclic encoding)
        'time_hour_sin', 'time_hour_cos', 'is_peak_hour',
        
        # Network Load (3 features - added load_per_user)
        'network_utilization', 'total_active_users', 'load_per_user',
        
        # Per-Slice Demands (3 features)
        'total_demand_embb', 'total_demand_urllc', 'total_demand_mmtc',
        
        # Demand Ratios (3 features)
        'embb_demand_ratio', 'urllc_demand_ratio', 'mmtc_demand_ratio',
        
        # Per-Slice SINR (3 features)
        'avg_sinr_embb', 'avg_sinr_urllc', 'avg_sinr_mmtc',
        
        # Per-Slice CQI (3 features)
        'avg_cqi_embb', 'avg_cqi_urllc', 'avg_cqi_mmtc',
        
        # Channel Quality (3 features)
        'embb_channel_quality', 'urllc_channel_quality', 'mmtc_channel_quality',
        
        # Slice Utilizations (3 features)
        'embb_utilization', 'urllc_utilization', 'mmtc_utilization',
        
        # Utilization-Demand Ratios (3 features)
        'embb_util_demand_ratio', 'urllc_util_demand_ratio', 'mmtc_util_demand_ratio',
    ]
    
    # Temporal features (deltas + moving averages)
    temporal_features = []
    for col in ['total_demand_embb', 'total_demand_urllc', 'total_demand_mmtc',
                'network_utilization', 'total_active_users']:
        temporal_features.extend([f'{col}_delta', f'{col}_ma3'])
    
    return base_features + temporal_features


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


def analyze_target_distribution(df, target_cols):
    """
    Analyze target distribution to inform normalization strategy
    """
    print("\n" + "=" * 80)
    print("TARGET DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    for col in target_cols:
        values = df[col].values
        print(f"\n{col}:")
        print(f"  Min:     {values.min():.4f}")
        print(f"  Max:     {values.max():.4f}")
        print(f"  Mean:    {values.mean():.4f}")
        print(f"  Median:  {np.median(values):.4f}")
        print(f"  Std:     {values.std():.4f}")
        print(f"  25th %:  {np.percentile(values, 25):.4f}")
        print(f"  75th %:  {np.percentile(values, 75):.4f}")
        print(f"  Zeros:   {(values == 0).sum()} ({(values == 0).mean()*100:.1f}%)")


def normalize_features(train_df, val_df, test_df, save_dir):
    """
    Normalize features using RobustScaler (better for outliers)
    Normalize targets with log1p + StandardScaler (handles wide range)
    """
    print("\n" + "=" * 80)
    print("FEATURE NORMALIZATION (RobustScaler - Outlier Robust)")
    print("=" * 80)
    
    feature_cols = get_feature_columns()
    target_cols = [
        'allocated_bandwidth_mbps_embb',
        'allocated_bandwidth_mbps_urllc',
        'allocated_bandwidth_mbps_mmtc'
    ]
    
    # Use RobustScaler for features (less sensitive to outliers than StandardScaler)
    X_scaler = RobustScaler()
    X_scaler.fit(train_df[feature_cols])
    
    print(f"Feature scaler fitted on {len(feature_cols)} features")
    print(f"  Median (center): {X_scaler.center_[:5]}... (first 5)")
    print(f"  IQR (scale):     {X_scaler.scale_[:5]}... (first 5)")
    
    # Transform all splits
    train_df[feature_cols] = X_scaler.transform(train_df[feature_cols])
    val_df[feature_cols] = X_scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = X_scaler.transform(test_df[feature_cols])
    
    # Verify normalization
    print(f"\nFeature ranges after normalization:")
    print(f"  Min:  {train_df[feature_cols].min().min():.2f}")
    print(f"  Max:  {train_df[feature_cols].max().max():.2f}")
    print(f"  Mean: {train_df[feature_cols].mean().mean():.2f}")
    
    # === TARGET NORMALIZATION ===
    print("\n" + "=" * 80)
    print("TARGET NORMALIZATION (Log1p + StandardScaler)")
    print("=" * 80)
    
    # Analyze raw target distribution
    analyze_target_distribution(train_df, target_cols)
    
    # Apply log1p transform (handles zeros/small values)
    train_targets_log = np.log1p(train_df[target_cols].values)
    val_targets_log = np.log1p(val_df[target_cols].values)
    test_targets_log = np.log1p(test_df[target_cols].values)
    
    print("\n" + "=" * 80)
    print("After log1p transform:")
    print("=" * 80)
    print(f"  Min:  {train_targets_log.min(axis=0)}")
    print(f"  Max:  {train_targets_log.max(axis=0)}")
    print(f"  Mean: {train_targets_log.mean(axis=0)}")
    
    # Fit StandardScaler on log-transformed targets
    y_scaler = StandardScaler()
    y_scaler.fit(train_targets_log)
    
    print(f"\nTarget scaler statistics:")
    print(f"  Mean: {y_scaler.mean_}")
    print(f"  Std:  {y_scaler.scale_}")
    
    # Transform all splits
    train_df[target_cols] = y_scaler.transform(train_targets_log)
    val_df[target_cols] = y_scaler.transform(val_targets_log)
    test_df[target_cols] = y_scaler.transform(test_targets_log)
    
    print("\n" + "=" * 80)
    print("After log1p + StandardScaler:")
    print("=" * 80)
    print(f"  Min:  {train_df[target_cols].min().values}")
    print(f"  Max:  {train_df[target_cols].max().values}")
    print(f"  Mean: {train_df[target_cols].mean().values}")
    print(f"  Std:  {train_df[target_cols].std().values}")
    
    # Save scalers
    joblib.dump(X_scaler, os.path.join(save_dir, 'feature_scaler.pkl'))
    joblib.dump(y_scaler, os.path.join(save_dir, 'target_scaler.pkl'))
    print(f"\n✓ Scalers saved to {save_dir}")
    
    return train_df, val_df, test_df, X_scaler, y_scaler


def verify_data_quality(train_df, val_df, test_df):
    """
    Verify no NaN/Inf values and reasonable ranges
    """
    print("\n" + "=" * 80)
    print("DATA QUALITY VERIFICATION")
    print("=" * 80)
    
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n{name} set:")
        
        # Check for NaN
        nan_count = df.isnull().sum().sum()
        print(f"  NaN values: {nan_count}")
        if nan_count > 0:
            print(f"  ⚠ WARNING: Found {nan_count} NaN values!")
            print(f"  Columns with NaN: {df.columns[df.isnull().any()].tolist()}")
        
        # Check for Inf
        inf_count = np.isinf(df.select_dtypes(include=[np.number]).values).sum()
        print(f"  Inf values: {inf_count}")
        if inf_count > 0:
            print(f"  ⚠ WARNING: Found {inf_count} Inf values!")
        
        # Check value ranges
        numeric_df = df.select_dtypes(include=[np.number])
        print(f"  Value range: [{numeric_df.min().min():.2f}, {numeric_df.max().max():.2f}]")
        
        if nan_count == 0 and inf_count == 0:
            print(f"  ✓ Data quality check passed")


def main():
    """
    Main preprocessing pipeline with improvements
    """
    print("\n" + "=" * 80)
    print("6G BANDWIDTH ALLOCATION - IMPROVED PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Paths
    raw_data_path = r"D:\Academics\SEM-5\Machine Learning\ML_courseproj\Dataset\nr_dataset_docker.csv"
    save_dir = r"D:\Academics\SEM-5\Machine Learning\ML_courseproj\Processed"
    os.makedirs(save_dir, exist_ok=True)
    
    # Load raw data
    print(f"\nLoading data from: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Aggregate per-UE to per-timestep
    df_agg = aggregate_per_timestep(df)
    
    # Engineer features
    df_agg = engineer_features(df_agg)
    
    # Split by episodes
    train_episodes = list(range(0, 80))
    val_episodes = list(range(80, 90))
    test_episodes = list(range(90, 100))
    
    train_df, val_df, test_df = split_episodes(df_agg, train_episodes, val_episodes, test_episodes)
    
    # Normalize features and targets
    train_df, val_df, test_df, X_scaler, y_scaler = normalize_features(
        train_df, val_df, test_df, save_dir
    )
    
    # Verify data quality
    verify_data_quality(train_df, val_df, test_df)
    
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
    
    # Save feature names for reference
    feature_cols = get_feature_columns()
    with open(os.path.join(save_dir, 'feature_names.txt'), 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f"✓ Saved feature_names.txt ({len(feature_cols)} features)")
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"Total features: {len(feature_cols)} (improved from 17)")
    print("Next step: Run sequence generation")
    print("=" * 80)


if __name__ == "__main__":
    main()