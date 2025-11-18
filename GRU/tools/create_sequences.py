"""
Create sequences from processed data for GRU training - FIXED VERSION
Updates: Uses new feature set, adds data validation
"""

import pandas as pd
import numpy as np
import os


def get_feature_columns():
    """
    MUST match features from preprocessing
    """
    base_features = [
        # Temporal (3 features)
        'time_hour_sin', 'time_hour_cos', 'is_peak_hour',
        
        # Network Load (3 features)
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


def create_sequences_per_episode(df, sequence_length=10):
    """
    Create sequences from timesteps within each episode
    
    Args:
        df: DataFrame with columns [episode_id, step_id, features..., targets...]
        sequence_length: Number of past timesteps to use (default: 10)
    
    Returns:
        X: Input sequences (N, seq_len, n_features)
        y: Target values (N, n_outputs)
    """
    
    feature_cols = get_feature_columns()
    
    target_cols = [
        'allocated_bandwidth_mbps_embb',
        'allocated_bandwidth_mbps_urllc',
        'allocated_bandwidth_mbps_mmtc'
    ]
    
    # Verify all features exist
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataframe: {missing_features}")
    
    print(f"Using {len(feature_cols)} features per timestep")
    
    X_sequences = []
    y_targets = []
    
    # Process each episode independently
    for episode_id in sorted(df['episode_id'].unique()):
        episode_df = df[df['episode_id'] == episode_id].sort_values('step_id')
        
        # Extract features and targets
        features = episode_df[feature_cols].values  # (timesteps, n_features)
        targets = episode_df[target_cols].values    # (timesteps, 3)
        
        # Verify no NaN/Inf
        if np.isnan(features).any() or np.isinf(features).any():
            print(f"⚠ WARNING: Episode {episode_id} contains NaN/Inf in features")
            continue
        
        # Create sliding windows
        for i in range(sequence_length, len(features)):
            # Input: past 'sequence_length' timesteps
            X_seq = features[i - sequence_length:i]  # (seq_len, n_features)
            
            # Target: allocation at current timestep
            y_target = targets[i]  # (3,)
            
            X_sequences.append(X_seq)
            y_targets.append(y_target)
    
    X = np.array(X_sequences, dtype=np.float32)  # (N, seq_len, n_features)
    y = np.array(y_targets, dtype=np.float32)    # (N, 3)
    
    return X, y


def verify_sequences(X, y, split_name):
    """
    Verify sequence quality
    """
    print(f"\nVerifying {split_name} sequences:")
    
    # Check shapes
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    
    # Check for NaN/Inf
    nan_count_X = np.isnan(X).sum()
    inf_count_X = np.isinf(X).sum()
    nan_count_y = np.isnan(y).sum()
    inf_count_y = np.isinf(y).sum()
    
    print(f"  X - NaN: {nan_count_X}, Inf: {inf_count_X}")
    print(f"  y - NaN: {nan_count_y}, Inf: {inf_count_y}")
    
    if nan_count_X > 0 or inf_count_X > 0 or nan_count_y > 0 or inf_count_y > 0:
        raise ValueError(f"⚠ ERROR: Found NaN/Inf in {split_name} sequences!")
    
    # Check value ranges
    print(f"  X range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"  y range: [{y.min():.2f}, {y.max():.2f}]")
    
    print(f"  ✓ Quality check passed")


def main():
    """
    Generate sequences for all splits
    """
    print("\n" + "=" * 80)
    print("SEQUENCE GENERATION FOR GRU - FIXED VERSION")
    print("=" * 80)
    
    # Paths
    data_dir = r"D:\Academics\SEM-5\Machine Learning\ML_courseproj\Processed"
    output_dir = r"D:\Academics\SEM-5\Machine Learning\ML_courseproj\SplitsMe"
    os.makedirs(output_dir, exist_ok=True)
    
    # Sequence parameters
    sequence_length = 10  # Use past 10 timesteps
    
    # Load processed data
    print("\nLoading processed data...")
    train_df = pd.read_csv(os.path.join(data_dir, 'train_processed.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val_processed.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_processed.csv'))
    
    print(f"Train: {len(train_df)} timesteps")
    print(f"Val:   {len(val_df)} timesteps")
    print(f"Test:  {len(test_df)} timesteps")
    
    # Generate sequences
    print("\n" + "=" * 80)
    print(f"CREATING SEQUENCES (sequence_length={sequence_length})")
    print("=" * 80)
    
    print("\n[1/3] Training set...")
    X_train, y_train = create_sequences_per_episode(train_df, sequence_length)
    verify_sequences(X_train, y_train, "Training")
    
    print("\n[2/3] Validation set...")
    X_val, y_val = create_sequences_per_episode(val_df, sequence_length)
    verify_sequences(X_val, y_val, "Validation")
    
    print("\n[3/3] Test set...")
    X_test, y_test = create_sequences_per_episode(test_df, sequence_length)
    verify_sequences(X_test, y_test, "Test")
    
    # Save sequences as .npy files
    print("\n" + "=" * 80)
    print("SAVING SEQUENCES")
    print("=" * 80)
    
    np.save(os.path.join(output_dir, 'X_train_lstm.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train_lstm.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_val_lstm.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_val_lstm.npy'), y_val)
    np.save(os.path.join(output_dir, 'X_test_lstm.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test_lstm.npy'), y_test)
    
    print(f"✓ Saved to: {output_dir}")
    print(f"  - X_train_lstm.npy: {X_train.shape} ({X_train.nbytes / 1e6:.2f} MB)")
    print(f"  - y_train_lstm.npy: {y_train.shape} ({y_train.nbytes / 1e6:.2f} MB)")
    print(f"  - X_val_lstm.npy:   {X_val.shape} ({X_val.nbytes / 1e6:.2f} MB)")
    print(f"  - y_val_lstm.npy:   {y_val.shape} ({y_val.nbytes / 1e6:.2f} MB)")
    print(f"  - X_test_lstm.npy:  {X_test.shape} ({X_test.nbytes / 1e6:.2f} MB)")
    print(f"  - y_test_lstm.npy:  {y_test.shape} ({y_test.nbytes / 1e6:.2f} MB)")
    
    # Summary
    print("\n" + "=" * 80)
    print("SEQUENCE GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nSequence format:")
    print(f"  Input:  (batch, {sequence_length}, {X_train.shape[2]}) - Past {sequence_length} timesteps")
    print(f"  Output: (batch, 3) - Bandwidth allocation for 3 slices")
    print(f"\n✓ Ready for GRU training with improved features!")
    print("=" * 80)


if __name__ == "__main__":
    main()