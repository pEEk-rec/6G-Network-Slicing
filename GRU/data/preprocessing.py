"""
Multi-Slice Data Preprocessor for 6G Bandwidth Allocation

Transforms per-slice data format to per-timestep format with multi-output targets.
Handles feature engineering, normalization, and data validation.

Key Features:
- Reshapes: 3 rows/timestep (per-slice) → 1 row/timestep (multi-output)
- Extracts per-slice features (eMBB, URLLC, mMTC specific)
- Normalizes using StandardScaler (fit on train only - prevents leakage)
- Validates data integrity throughout pipeline

Author: 6G Bandwidth Allocation Project
Date: 2024
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MultiSlicePreprocessor:
    """
    Preprocessor for multi-slice 6G network data.
    
    Transforms data from per-slice format (3 rows per timestep) to 
    per-timestep format (1 row with features from all 3 slices).
    
    Features Strategy:
    - Shared features: network_utilization, time_hour, is_peak_hour, etc.
    - Per-slice features: demand, SINR, CQI, etc. for each slice
    - Total: ~25-30 features preserving per-slice information
    
    Target Variables:
    - allocated_prbs_embb: PRBs allocated to eMBB slice
    - allocated_prbs_urllc: PRBs allocated to URLLC slice  
    - allocated_prbs_mmtc: PRBs allocated to mMTC slice
    
    Example
    -------
    >>> preprocessor = MultiSlicePreprocessor()
    >>> train_processed = preprocessor.fit_transform(train_df)
    >>> val_processed = preprocessor.transform(val_df)
    >>> preprocessor.save_scaler('models/scaler.pkl')
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_columns = ['prb_embb', 'prb_urllc', 'prb_mmtc']
        self.is_fitted = False
        self.stats = {}
        
        # Feature groups for organization
        self.SHARED_FEATURES = [
            'network_utilization',
            'time_hour', 
            'is_peak_hour',
            'total_active_users'
        ]
        
        self.PER_SLICE_FEATURES = [
            'total_demand',
            'sinr_db',
            'rsrp_dbm',
            'cqi',
            'distance_to_gnb',
            'num_users'
        ]
        
        # Cross-slice awareness features
        self.CROSS_SLICE_FEATURES = [
            'embb_utilization',
            'urllc_utilization',
            'mmtc_utilization'
        ]
    
    
    def reshape_per_timestep(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reshape from per-slice format to per-timestep format.
        
        Input format:  3 rows per timestep (one per slice)
        Output format: 1 row per timestep (all slice info combined)
        
        Parameters
        ----------
        df : pd.DataFrame
            Data in per-slice format with 'slice_type' column
            
        Returns
        -------
        pd.DataFrame
            Reshaped data with one row per timestep
        """
        print(f"\n{'='*70}")
        print(f"RESHAPING DATA: Per-Slice → Per-Timestep")
        print(f"{'='*70}")
        print(f"Input shape: {df.shape}")
        print(f"Input format: {len(df)} rows (3 rows per timestep)")
        
        # Verify we have all 3 slices
        slice_counts = df['slice_type'].value_counts().sort_index()
        print(f"\nSlice distribution:")
        for slice_id, count in slice_counts.items():
            slice_name = {1: 'eMBB', 2: 'URLLC', 3: 'mMTC'}.get(slice_id, 'Unknown')
            print(f"  Slice {slice_id} ({slice_name}): {count:,} rows")
        
        if len(slice_counts) != 3:
            raise ValueError(f"Expected 3 slices, found {len(slice_counts)}")
        
        # Group by episode and timestep
        grouped = df.groupby(['episode_id', 'step_id'])
        
        # Verify each group has exactly 3 rows
        group_sizes = grouped.size()
        if not (group_sizes == 3).all():
            bad_groups = group_sizes[group_sizes != 3]
            print(f"\n⚠️ WARNING: {len(bad_groups)} timesteps don't have 3 slices!")
            print(f"First few problematic groups:")
            print(bad_groups.head())
            raise ValueError("Some timesteps missing slice data")
        
        print(f"\n✓ All timesteps have 3 slices")
        print(f"Creating {len(grouped)} timestep rows...")
        
        # Extract data for each slice
        timestep_data = []
        
        for (episode_id, step_id), group in grouped:
            # Get data for each slice (sorted by slice_type to ensure order)
            group_sorted = group.sort_values('slice_type')
            
            if len(group_sorted) != 3:
                continue
            
            embb_row = group_sorted[group_sorted['slice_type'] == 1].iloc[0]
            urllc_row = group_sorted[group_sorted['slice_type'] == 2].iloc[0]
            mmtc_row = group_sorted[group_sorted['slice_type'] == 3].iloc[0]
            
            # Build timestep row
            timestep_row = {
                'episode_id': episode_id,
                'step_id': step_id,
                
                # Shared features (same across all slices at this timestep)
                'network_utilization': embb_row['network_utilization'],
                'time_hour': embb_row['time_hour'],
                'is_peak_hour': embb_row['is_peak_hour'],
                'total_active_users': embb_row['total_active_users'],
                'timestamp': embb_row['timestamp'],
                
                # Cross-slice utilizations
                'embb_utilization': embb_row['embb_utilization'],
                'urllc_utilization': embb_row['urllc_utilization'],
                'mmtc_utilization': embb_row['mmtc_utilization'],
                
                # eMBB-specific features
                'embb_demand': embb_row['total_demand'],
                'embb_sinr': embb_row['sinr_db'],
                'embb_rsrp': embb_row['rsrp_dbm'],
                'embb_cqi': embb_row['cqi'],
                'embb_distance': embb_row['distance_to_gnb'],
                'embb_num_users': embb_row['num_users'],
                
                # URLLC-specific features
                'urllc_demand': urllc_row['total_demand'],
                'urllc_sinr': urllc_row['sinr_db'],
                'urllc_rsrp': urllc_row['rsrp_dbm'],
                'urllc_cqi': urllc_row['cqi'],
                'urllc_distance': urllc_row['distance_to_gnb'],
                'urllc_num_users': urllc_row['num_users'],
                
                # mMTC-specific features
                'mmtc_demand': mmtc_row['total_demand'],
                'mmtc_sinr': mmtc_row['sinr_db'],
                'mmtc_rsrp': mmtc_row['rsrp_dbm'],
                'mmtc_cqi': mmtc_row['cqi'],
                'mmtc_distance': mmtc_row['distance_to_gnb'],
                'mmtc_num_users': mmtc_row['num_users'],
                
                # Target variables (PRB allocations)
                'prb_embb': embb_row['allocated_prbs'],
                'prb_urllc': urllc_row['allocated_prbs'],
                'prb_mmtc': mmtc_row['allocated_prbs'],
                
                # Additional useful info (not used as features)
                'embb_latency': embb_row['actual_latency_ms'],
                'urllc_latency': urllc_row['actual_latency_ms'],
                'mmtc_latency': mmtc_row['actual_latency_ms'],
                'embb_qos_violated': embb_row['qos_violated'],
                'urllc_qos_violated': urllc_row['qos_violated'],
                'mmtc_qos_violated': mmtc_row['qos_violated'],
            }
            
            timestep_data.append(timestep_row)
        
        # Create DataFrame
        reshaped_df = pd.DataFrame(timestep_data)
        
        print(f"\n✓ Reshaping complete!")
        print(f"Output shape: {reshaped_df.shape}")
        print(f"Output format: {len(reshaped_df)} rows (1 row per timestep)")
        print(f"Reduction: {len(df)} → {len(reshaped_df)} rows ({len(df)//len(reshaped_df)}:1)")
        print(f"{'='*70}")
        
        return reshaped_df
    
    
    def extract_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract feature matrix and target matrix.
        
        Parameters
        ----------
        df : pd.DataFrame
            Reshaped data (one row per timestep)
            
        Returns
        -------
        X : pd.DataFrame
            Feature matrix (25 features)
        y : pd.DataFrame
            Target matrix (3 targets: prb_embb, prb_urllc, prb_mmtc)
        """
        # Define feature columns
        feature_cols = (
            self.SHARED_FEATURES +
            self.CROSS_SLICE_FEATURES +
            [f'embb_{feat}' for feat in ['demand', 'sinr', 'rsrp', 'cqi', 'distance', 'num_users']] +
            [f'urllc_{feat}' for feat in ['demand', 'sinr', 'rsrp', 'cqi', 'distance', 'num_users']] +
            [f'mmtc_{feat}' for feat in ['demand', 'sinr', 'rsrp', 'cqi', 'distance', 'num_users']]
        )
        
        # Verify all features exist
        missing_features = [f for f in feature_cols if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Extract features and targets
        X = df[feature_cols].copy()
        y = df[self.target_columns].copy()
        
        # Store feature column names
        self.feature_columns = feature_cols
        
        return X, y
    
    
    def fit_transform(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fit preprocessor on training data and transform.
        
        CRITICAL: This method ONLY sees training data.
        Scaler statistics (mean, std) are computed ONLY from training set.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data in per-slice format
            
        Returns
        -------
        X_train : pd.DataFrame
            Normalized feature matrix
        y_train : pd.DataFrame
            Target matrix (PRB allocations)
        metadata : pd.DataFrame
            Metadata (episode_id, step_id, timestamps, etc.)
        """
        print(f"\n{'='*70}")
        print(f"FIT_TRANSFORM: Training Data Preprocessing")
        print(f"{'='*70}")
        
        # Step 1: Reshape
        train_reshaped = self.reshape_per_timestep(train_df)
        
        # Step 2: Extract features and targets
        print(f"\nExtracting features and targets...")
        X_train, y_train = self.extract_features(train_reshaped)
        
        print(f"✓ Feature matrix: {X_train.shape}")
        print(f"✓ Target matrix: {y_train.shape}")
        print(f"\nFeatures ({len(self.feature_columns)}):")
        for i, feat in enumerate(self.feature_columns, 1):
            print(f"  {i:2d}. {feat}")
        
        # Step 3: Fit and transform scaler
        print(f"\n{'='*70}")
        print(f"NORMALIZATION: Fitting StandardScaler on Training Data")
        print(f"{'='*70}")
        
        print(f"Computing statistics (mean, std) from training data...")
        self.scaler.fit(X_train)
        X_train_normalized = pd.DataFrame(
            self.scaler.transform(X_train),
            columns=self.feature_columns,
            index=X_train.index
        )
        
        print(f"✓ Scaler fitted on {len(X_train):,} training samples")
        
        # Store statistics
        self.stats['feature_means'] = dict(zip(self.feature_columns, self.scaler.mean_))
        self.stats['feature_stds'] = dict(zip(self.feature_columns, self.scaler.scale_))
        
        print(f"\nFeature statistics (training data):")
        print(f"{'Feature':<30} {'Mean':>12} {'Std':>12}")
        print(f"{'-'*54}")
        for feat in self.feature_columns[:10]:  # Show first 10
            mean = self.stats['feature_means'][feat]
            std = self.stats['feature_stds'][feat]
            print(f"{feat:<30} {mean:>12.3f} {std:>12.3f}")
        print(f"... ({len(self.feature_columns)-10} more features)")
        
        # Store metadata
        metadata_cols = ['episode_id', 'step_id', 'timestamp']
        metadata = train_reshaped[metadata_cols].copy()
        
        self.is_fitted = True
        print(f"\n✓ Preprocessing complete for training data")
        print(f"{'='*70}")
        
        return X_train_normalized, y_train, metadata
    
    
    def transform(self, df: pd.DataFrame, split_name: str = 'validation') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Transform validation/test data using fitted scaler.
        
        CRITICAL: Uses statistics from TRAINING data only.
        Does NOT recompute mean/std - this prevents data leakage!
        
        Parameters
        ----------
        df : pd.DataFrame
            Data in per-slice format
        split_name : str
            'validation' or 'test' (for logging)
            
        Returns
        -------
        X : pd.DataFrame
            Normalized feature matrix
        y : pd.DataFrame
            Target matrix
        metadata : pd.DataFrame
            Metadata columns
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted! Call fit_transform() on training data first.")
        
        print(f"\n{'='*70}")
        print(f"TRANSFORM: {split_name.upper()} Data Preprocessing")
        print(f"{'='*70}")
        
        # Step 1: Reshape
        reshaped = self.reshape_per_timestep(df)
        
        # Step 2: Extract features and targets
        print(f"\nExtracting features and targets...")
        X, y = self.extract_features(reshaped)
        
        print(f"✓ Feature matrix: {X.shape}")
        print(f"✓ Target matrix: {y.shape}")
        
        # Step 3: Transform using fitted scaler (NO FITTING!)
        print(f"\n{'='*70}")
        print(f"NORMALIZATION: Applying Training Statistics")
        print(f"{'='*70}")
        print(f"Using mean/std from TRAINING data (prevents data leakage)")
        
        X_normalized = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_columns,
            index=X.index
        )
        
        print(f"✓ Normalized {len(X):,} samples using training statistics")
        
        # Store metadata
        metadata_cols = ['episode_id', 'step_id', 'timestamp']
        metadata = reshaped[metadata_cols].copy()
        
        print(f"\n✓ Preprocessing complete for {split_name} data")
        print(f"{'='*70}")
        
        return X_normalized, y, metadata
    
    
    def save_scaler(self, filepath: str):
        """Save fitted scaler for later use."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted yet!")
        
        scaler_data = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'stats': self.stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(scaler_data, f)
        
        print(f"\n✓ Scaler saved to: {filepath}")
    
    
    def load_scaler(self, filepath: str):
        """Load previously fitted scaler."""
        with open(filepath, 'rb') as f:
            scaler_data = pickle.load(f)
        
        self.scaler = scaler_data['scaler']
        self.feature_columns = scaler_data['feature_columns']
        self.target_columns = scaler_data['target_columns']
        self.stats = scaler_data['stats']
        self.is_fitted = True
        
        print(f"\n✓ Scaler loaded from: {filepath}")
        print(f"✓ {len(self.feature_columns)} features, {len(self.target_columns)} targets")
    
    
    def get_preprocessing_info(self) -> Dict:
        """Get preprocessing configuration and statistics."""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted yet!")
        
        return {
            'num_features': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'num_targets': len(self.target_columns),
            'target_columns': self.target_columns,
            'feature_statistics': self.stats,
            'normalization': 'StandardScaler (z-score)'
        }


def preprocess_all_splits(
    train_path: str,
    val_path: str,
    test_path: str,
    output_dir: str,
    save_scaler_path: Optional[str] = None
) -> Dict:
    """
    Convenience function to preprocess all splits at once.
    
    Parameters
    ----------
    train_path : str
        Path to train.csv
    val_path : str
        Path to val.csv
    test_path : str
        Path to test.csv
    output_dir : str
        Directory to save preprocessed files
    save_scaler_path : str, optional
        Path to save fitted scaler
        
    Returns
    -------
    dict
        Preprocessing summary and file paths
    """
    print(f"\n{'='*70}")
    print(f"MULTI-SLICE PREPROCESSING PIPELINE")
    print(f"{'='*70}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = MultiSlicePreprocessor()
    
    # Load data
    print(f"\nLoading data...")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Test:  {test_path}")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    print(f"\n✓ Data loaded:")
    print(f"  Train: {len(train_df):,} rows")
    print(f"  Val:   {len(val_df):,} rows")
    print(f"  Test:  {len(test_df):,} rows")
    
    # Process training data
    X_train, y_train, meta_train = preprocessor.fit_transform(train_df)
    
    # Process validation data
    X_val, y_val, meta_val = preprocessor.transform(val_df, 'validation')
    
    # Process test data
    X_test, y_test, meta_test = preprocessor.transform(test_df, 'test')
    
    # Combine and save
    print(f"\n{'='*70}")
    print(f"SAVING PREPROCESSED DATA")
    print(f"{'='*70}")
    
    train_processed = pd.concat([meta_train, X_train, y_train], axis=1)
    val_processed = pd.concat([meta_val, X_val, y_val], axis=1)
    test_processed = pd.concat([meta_test, X_test, y_test], axis=1)
    
    train_output = output_path / 'train_processed.csv'
    val_output = output_path / 'val_processed.csv'
    test_output = output_path / 'test_processed.csv'
    
    print(f"\nSaving preprocessed files...")
    train_processed.to_csv(train_output, index=False)
    print(f"  → {train_output}")
    
    val_processed.to_csv(val_output, index=False)
    print(f"  → {val_output}")
    
    test_processed.to_csv(test_output, index=False)
    print(f"  → {test_output}")
    
    # Save scaler
    if save_scaler_path:
        preprocessor.save_scaler(save_scaler_path)
    
    # Save preprocessing info
    info_path = output_path / 'preprocessing_info.json'
    info = preprocessor.get_preprocessing_info()
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"\nSaving preprocessing info...")
    print(f"  → {info_path}")
    
    print(f"\n{'='*70}")
    print(f"✓ PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"\nSummary:")
    print(f"  Train: {len(train_processed):,} timesteps, {train_processed.shape[1]} columns")
    print(f"  Val:   {len(val_processed):,} timesteps, {val_processed.shape[1]} columns")
    print(f"  Test:  {len(test_processed):,} timesteps, {test_processed.shape[1]} columns")
    print(f"\nFeatures: {len(X_train.columns)}")
    print(f"Targets:  {len(y_train.columns)} (prb_embb, prb_urllc, prb_mmtc)")
    print(f"\nNEXT STEPS:")
    print(f"  1. Generate sequences: python data/sequence_generator.py")
    print(f"  2. Train model: python experiments/train_gru.py")
    print(f"{'='*70}\n")
    
    return {
        'train_output': str(train_output),
        'val_output': str(val_output),
        'test_output': str(test_output),
        'num_features': len(X_train.columns),
        'num_targets': len(y_train.columns),
        'train_samples': len(train_processed),
        'val_samples': len(val_processed),
        'test_samples': len(test_processed)
    }


if __name__ == "__main__":
    """
    Command-line usage:
    python data/preprocessor.py
    """
    import sys
    
    # Default paths (Windows-compatible)
    default_train = r'D:\Academics\SEM-5\Machine Learning\ML_courseproj\Splits\train.csv'
    default_val = r'D:\Academics\SEM-5\Machine Learning\ML_courseproj\Splits\val.csv'
    default_test = r'D:\Academics\SEM-5\Machine Learning\ML_courseproj\Splits\test.csv'
    default_output = r'D:\Academics\SEM-5\Machine Learning\ML_courseproj\Processed'
    default_scaler = r'D:\Academics\SEM-5\Machine Learning\ML_courseproj\Processed\scaler.pkl'
    
    print("\n" + "="*70)
    print("6G BANDWIDTH ALLOCATION - DATA PREPROCESSOR")
    print("="*70)
    
    if len(sys.argv) > 3:
        train_path = sys.argv[1]
        val_path = sys.argv[2]
        test_path = sys.argv[3]
        output_dir = sys.argv[4] if len(sys.argv) > 4 else default_output
    else:
        print(f"\nUsage: python data/preprocessor.py <train.csv> <val.csv> <test.csv> [output_dir]")
        print(f"\nUsing default paths:")
        print(f"  Train:  {default_train}")
        print(f"  Val:    {default_val}")
        print(f"  Test:   {default_test}")
        print(f"  Output: {default_output}")
        
        train_path = default_train
        val_path = default_val
        test_path = default_test
        output_dir = default_output
    
    # Check if files exist
    from pathlib import Path
    for path in [train_path, val_path, test_path]:
        if not Path(path).exists():
            print(f"\n Error: File not found: {path}")
            sys.exit(1)
    
    # Run preprocessing
    try:
        summary = preprocess_all_splits(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            output_dir=output_dir,
            save_scaler_path=default_scaler
        )
        
        print(f"\n✓ SUCCESS! Preprocessed data ready for sequence generation.")
        
    except Exception as e:
        print(f"\n Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)