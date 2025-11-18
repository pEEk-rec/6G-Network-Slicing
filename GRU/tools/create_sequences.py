"""
Create sequences from processed data for GRU training
Converts per-timestep data → sliding window sequences
"""

import pandas as pd
import numpy as np
import os


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
    
    X_sequences = []
    y_targets = []
    
    # Process each episode independently
    for episode_id in df['episode_id'].unique():
        episode_df = df[df['episode_id'] == episode_id].sort_values('step_id')
        
        # Extract features and targets
        features = episode_df[feature_cols].values  # (timesteps, 17)
        targets = episode_df[target_cols].values    # (timesteps, 3)
        
        # Create sliding windows
        for i in range(sequence_length, len(features)):
            # Input: past 'sequence_length' timesteps
            X_seq = features[i - sequence_length:i]  # (10, 17)
            
            # Target: allocation at current timestep
            y_target = targets[i]  # (3,)
            
            X_sequences.append(X_seq)
            y_targets.append(y_target)
    
    X = np.array(X_sequences)  # (N, 10, 17)
    y = np.array(y_targets)    # (N, 3)
    
    return X, y


def main():
    """
    Generate sequences for all splits
    """
    print("\n" + "=" * 80)
    print("SEQUENCE GENERATION FOR GRU")
    print("=" * 80)
    
    # Paths
    data_dir = r"D:\Academics\SEM-5\Machine Learning\ML_courseproj\Processed"
    output_dir = "D:/Academics/SEM-5/Machine Learning/ML_courseproj/SplitsMe"
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
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    
    print("\n[2/3] Validation set...")
    X_val, y_val = create_sequences_per_episode(val_df, sequence_length)
    print(f"  X_val:   {X_val.shape}")
    print(f"  y_val:   {y_val.shape}")
    
    print("\n[3/3] Test set...")
    X_test, y_test = create_sequences_per_episode(test_df, sequence_length)
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test.shape}")
    
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
    print(f"  Input:  (batch, {sequence_length}, 17) - Past {sequence_length} timesteps with 17 features each")
    print(f"  Output: (batch, 3) - Bandwidth allocation for 3 slices")
    print("\nReady for GRU training!")
    print("=" * 80)


if __name__ == "__main__":
    main()
