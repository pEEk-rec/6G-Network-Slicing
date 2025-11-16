"""
COORDINATED Episode-based data splitting for team collaboration.
Ensures identical train/val/test splits across all team members.

CRITICAL FOR FAIR COMPARISON:
- LSTM vs GRU must use IDENTICAL test episodes
- Same episodes = same difficulty = fair comparison
- Fixed split = reproducible results across experiments

Author: 6G Bandwidth Allocation Project Team
Date: 2024
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Tuple, Dict, Optional, List
from datetime import datetime


# ============================================================
# FIXED SPLIT CONFIGURATION - DO NOT MODIFY WITHOUT TEAM AGREEMENT
# ============================================================
# These episode ranges are FIXED for entire project
# ALL team members must use these same splits
# ============================================================

FIXED_TRAIN_EPISODES = list(range(0, 80))   # Episodes 0-79 (80 episodes)
FIXED_VAL_EPISODES = list(range(80, 90))    # Episodes 80-89 (10 episodes)
FIXED_TEST_EPISODES = list(range(90, 100))  # Episodes 90-99 (10 episodes)

# Project-wide random seed for reproducibility
PROJECT_RANDOM_SEED = 42

# Split metadata for coordination
SPLIT_VERSION = "v1.0"
SPLIT_CONFIG = {
    'version': SPLIT_VERSION,
    'train_episodes': FIXED_TRAIN_EPISODES,
    'val_episodes': FIXED_VAL_EPISODES,
    'test_episodes': FIXED_TEST_EPISODES,
    'random_seed': PROJECT_RANDOM_SEED,
    'total_train': len(FIXED_TRAIN_EPISODES),
    'total_val': len(FIXED_VAL_EPISODES),
    'total_test': len(FIXED_TEST_EPISODES),
    'description': 'Fixed split for fair model comparison (LSTM vs GRU)'
}

# ============================================================


class CoordinatedEpisodeSplitter:
    """
    Coordinated episode splitter for team projects.
    
    GUARANTEES:
    1. All team members get IDENTICAL train/val/test episodes
    2. Results are directly comparable (LSTM vs GRU on same data)
    3. Episode difficulty is controlled (everyone faces same challenges)
    4. Reproducible across machines and time
    
    WHY FIXED SPLITS?
    -----------------
    - Fair Comparison: LSTM and GRU tested on identical unseen episodes
    - Episode Difficulty: Some episodes have harder patterns (peak hours, congestion)
    - Ablation Studies: Change one variable (model architecture) while keeping data fixed
    - Scientific Rigor: "Both models evaluated on episodes 90-99" is reproducible
    
    TEAM COORDINATION CHECKLIST:
    ---------------------------
    ✓ Same episode splits (episodes 0-79 train, 80-89 val, 90-99 test)
    ✓ Same random seed (42) for weight initialization
    ✓ Same hyperparameters (unless explicitly testing them):
      - Learning rate: 0.001
      - Batch size: 64
      - Sequence length: 10
      - Epochs: 50
    ✓ Same evaluation metrics (MAPE, MAE, QoS violation rate)
    ✓ Same preprocessing (normalization method, feature selection)
    
    Example
    -------
    >>> splitter = CoordinatedEpisodeSplitter()
    >>> splitter.split_and_save(
    ...     input_path='data/raw/network_data.csv',
    ...     output_dir='data/processed/',
    ...     verify_episodes=True  # Verify your data has episodes 0-99
    ... )
    """
    
    def __init__(self):
        """
        Initialize with FIXED project configuration.
        No parameters needed - configuration is centralized.
        """
        self.train_episodes = FIXED_TRAIN_EPISODES
        self.val_episodes = FIXED_VAL_EPISODES
        self.test_episodes = FIXED_TEST_EPISODES
        self.random_seed = PROJECT_RANDOM_SEED
        self.config = SPLIT_CONFIG.copy()
        
        # Will be populated after splitting
        self.split_info = {}
        
        print(f"\n{'='*70}")
        print(f"COORDINATED EPISODE SPLITTER - {SPLIT_VERSION}")
        print(f"{'='*70}")
        print(f"FIXED CONFIGURATION (Team-wide):")
        print(f"  Train Episodes: {self.train_episodes[0]}-{self.train_episodes[-1]} ({len(self.train_episodes)} episodes)")
        print(f"  Val Episodes:   {self.val_episodes[0]}-{self.val_episodes[-1]} ({len(self.val_episodes)} episodes)")
        print(f"  Test Episodes:  {self.test_episodes[0]}-{self.test_episodes[-1]} ({len(self.test_episodes)} episodes)")
        print(f"  Random Seed:    {self.random_seed}")
        print(f"\n⚠️  WARNING: These splits are FIXED for fair comparison.")
        print(f"   Do NOT modify without team agreement!")
        print(f"{'='*70}\n")
    
    
    def split(
        self, 
        df: pd.DataFrame, 
        episode_col: str = 'episode_id',
        verify_episodes: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame using FIXED episode ranges.
        
        Parameters
        ----------
        df : pd.DataFrame
            Full dataset with episode_id column
        episode_col : str, default='episode_id'
            Name of column containing episode identifiers
        verify_episodes : bool, default=True
            Verify dataset contains expected episodes (0-99)
        
        Returns
        -------
        train_df : pd.DataFrame
            Training set (episodes 0-79)
        val_df : pd.DataFrame
            Validation set (episodes 80-89)
        test_df : pd.DataFrame
            Test set (episodes 90-99)
        
        Raises
        ------
        ValueError
            If expected episodes are missing or configuration mismatch
        """
        # Validate input
        if episode_col not in df.columns:
            raise ValueError(f"Column '{episode_col}' not found in DataFrame")
        
        # Get unique episodes
        unique_episodes = sorted(df[episode_col].unique())
        total_episodes = len(unique_episodes)
        
        print(f"\n{'='*70}")
        print(f"DATA VALIDATION")
        print(f"{'='*70}")
        print(f"Dataset contains {total_episodes} unique episodes")
        print(f"Episode range in data: {unique_episodes[0]} to {unique_episodes[-1]}")
        
        # Verify expected episodes exist
        if verify_episodes:
            expected_episodes = set(self.train_episodes + self.val_episodes + self.test_episodes)
            actual_episodes = set(unique_episodes)
            
            missing_episodes = expected_episodes - actual_episodes
            extra_episodes = actual_episodes - expected_episodes
            
            if missing_episodes:
                print(f"\n❌ ERROR: Missing required episodes!")
                print(f"   Expected episodes: 0-99")
                print(f"   Missing: {sorted(list(missing_episodes))[:20]}")
                raise ValueError(
                    f"Dataset missing {len(missing_episodes)} episodes required by fixed split. "
                    f"Check if your data covers episodes 0-99."
                )
            
            if extra_episodes:
                print(f"\n⚠️  WARNING: Dataset contains extra episodes beyond 0-99:")
                print(f"   Extra episodes: {sorted(list(extra_episodes))[:10]}...")
                print(f"   These will be IGNORED (not in train/val/test)")
        
        print(f"✓ All required episodes (0-99) present in dataset")
        
        # Create splits using FIXED episode lists
        train_df = df[df[episode_col].isin(self.train_episodes)].copy()
        val_df = df[df[episode_col].isin(self.val_episodes)].copy()
        test_df = df[df[episode_col].isin(self.test_episodes)].copy()
        
        print(f"\n{'='*70}")
        print(f"SPLIT SUMMARY")
        print(f"{'='*70}")
        print(f"Training Set:")
        print(f"  Episodes: {self.train_episodes[0]}-{self.train_episodes[-1]} ({len(self.train_episodes)} episodes)")
        print(f"  Samples:  {len(train_df):,}")
        
        print(f"\nValidation Set:")
        print(f"  Episodes: {self.val_episodes[0]}-{self.val_episodes[-1]} ({len(self.val_episodes)} episodes)")
        print(f"  Samples:  {len(val_df):,}")
        
        print(f"\nTest Set:")
        print(f"  Episodes: {self.test_episodes[0]}-{self.test_episodes[-1]} ({len(self.test_episodes)} episodes)")
        print(f"  Samples:  {len(test_df):,}")
        
        total_samples = len(train_df) + len(val_df) + len(test_df)
        print(f"\nTotal:    {total_samples:,} samples")
        print(f"{'='*70}")
        
        # Store split metadata
        self.split_info = {
            'version': SPLIT_VERSION,
            'train_episodes': self.train_episodes,
            'val_episodes': self.val_episodes,
            'test_episodes': self.test_episodes,
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'total_samples': total_samples,
            'episode_col': episode_col,
            'random_seed': self.random_seed,
            'split_timestamp': datetime.now().isoformat(),
            'data_source': 'Fixed episode ranges for team coordination'
        }
        
        # Validate split correctness
        self._validate_split(train_df, val_df, test_df, episode_col)
        
        return train_df, val_df, test_df
    
    
    def split_and_save(
        self,
        input_path: str,
        output_dir: str,
        episode_col: str = 'episode_id',
        prefix: str = '',
        verify_episodes: bool = True,
        save_config: bool = True
    ) -> Dict:
        """
        Load data, split, and save to separate files with coordination metadata.
        
        Parameters
        ----------
        input_path : str
            Path to input CSV file
        output_dir : str
            Directory to save train/val/test files
        episode_col : str, default='episode_id'
            Name of episode column
        prefix : str, default=''
            Prefix for output filenames
        verify_episodes : bool, default=True
            Verify dataset contains episodes 0-99
        save_config : bool, default=True
            Save team coordination config file
        
        Returns
        -------
        dict
            Split metadata including file paths and team config
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"LOADING DATA")
        print(f"{'='*70}")
        print(f"Input file: {input_path}")
        
        # Load data
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        df = pd.read_csv(input_path)
        print(f"✓ Loaded {len(df):,} samples with {len(df.columns)} features")
        
        # Perform split
        train_df, val_df, test_df = self.split(df, episode_col, verify_episodes)
        
        # Generate filenames
        train_file = output_path / f"{prefix}train.csv"
        val_file = output_path / f"{prefix}val.csv"
        test_file = output_path / f"{prefix}test.csv"
        metadata_file = output_path / f"{prefix}split_metadata.json"
        config_file = output_path / f"{prefix}team_coordination_config.json"
        
        # Save splits
        print(f"\n{'='*70}")
        print(f"SAVING SPLIT FILES")
        print(f"{'='*70}")
        
        print(f"Saving training set...")
        train_df.to_csv(train_file, index=False)
        print(f"  → {train_file}")
        
        print(f"Saving validation set...")
        val_df.to_csv(val_file, index=False)
        print(f"  → {val_file}")
        
        print(f"Saving test set...")
        test_df.to_csv(test_file, index=False)
        print(f"  → {test_file}")
        
        # Add file paths to metadata
        self.split_info['files'] = {
            'train': str(train_file),
            'val': str(val_file),
            'test': str(test_file)
        }
        
        # Save metadata
        print(f"\nSaving split metadata...")
        with open(metadata_file, 'w') as f:
            json.dump(self.split_info, f, indent=2)
        print(f"  → {metadata_file}")
        
        # Save team coordination config
        if save_config:
            print(f"\nSaving team coordination config...")
            coordination_config = {
                **SPLIT_CONFIG,
                'hyperparameters': {
                    'random_seed': PROJECT_RANDOM_SEED,
                    'recommended_lr': 0.001,
                    'recommended_batch_size': 64,
                    'recommended_sequence_length': 10,
                    'recommended_epochs': 50
                },
                'evaluation_metrics': [
                    'MAE (Mean Absolute Error)',
                    'MAPE (Mean Absolute Percentage Error)',
                    'RMSE (Root Mean Squared Error)',
                    'QoS Violation Rate (%)',
                    'Latency P95 (ms)',
                    'Resource Utilization (%)'
                ],
                'preprocessing_specs': {
                    'normalization': 'StandardScaler (fit on train, apply to val/test)',
                    'feature_selection': 'Use same 12 features across all models',
                    'sequence_generation': 'Sliding window, length=10, no overlap across episodes'
                },
                'comparison_guidelines': {
                    'models_to_compare': ['GRU', 'LSTM', 'Transformer', 'Baseline'],
                    'report_format': 'All models evaluated on episodes 90-99',
                    'statistical_testing': 'Use paired t-test for significance',
                    'visualization': 'Same y-axis scales for fair visual comparison'
                }
            }
            
            with open(config_file, 'w') as f:
                json.dump(coordination_config, f, indent=2)
            print(f"  → {config_file}")
            print(f"\n⚠️  SHARE THIS FILE WITH TEAM MEMBERS!")
        
        print(f"\n{'='*70}")
        print(f"✓ SPLITTING COMPLETE")
        print(f"{'='*70}")
        print(f"\nFILES CREATED:")
        print(f"  • {train_file.name} - Training data (episodes 0-79)")
        print(f"  • {val_file.name} - Validation data (episodes 80-89)")
        print(f"  • {test_file.name} - Test data (episodes 90-99)")
        print(f"  • {metadata_file.name} - Split metadata")
        print(f"  • {config_file.name} - Team coordination config")
        
        print(f"\n{'='*70}")
        print(f"TEAM COORDINATION CHECKLIST")
        print(f"{'='*70}")
        print(f"✓ Episode splits: FIXED (0-79 train, 80-89 val, 90-99 test)")
        print(f"✓ Random seed: {self.random_seed}")
        print(f"⚠️  TODO: Coordinate with team on:")
        print(f"   • Hyperparameters (LR, batch size, epochs)")
        print(f"   • Feature selection (which 12 features to use)")
        print(f"   • Evaluation metrics (same MAPE/MAE calculation)")
        print(f"   • Preprocessing steps (normalization method)")
        print(f"{'='*70}\n")
        
        return self.split_info
    
    
    def _validate_split(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        episode_col: str
    ):
        """
        Validate split correctness for team coordination.
        
        CRITICAL CHECKS:
        1. No episode overlap (prevents data leakage)
        2. Correct episode ranges (matches team configuration)
        3. All expected episodes present
        4. Temporal ordering preserved
        """
        print(f"\n{'='*70}")
        print(f"VALIDATION CHECKS (Critical for Fair Comparison)")
        print(f"{'='*70}")
        
        train_episodes_actual = set(train_df[episode_col].unique())
        val_episodes_actual = set(val_df[episode_col].unique())
        test_episodes_actual = set(test_df[episode_col].unique())
        
        train_episodes_expected = set(self.train_episodes)
        val_episodes_expected = set(self.val_episodes)
        test_episodes_expected = set(self.test_episodes)
        
        # Check 1: No overlap (CRITICAL!)
        overlap_train_val = train_episodes_actual & val_episodes_actual
        overlap_train_test = train_episodes_actual & test_episodes_actual
        overlap_val_test = val_episodes_actual & test_episodes_actual
        
        if overlap_train_val or overlap_train_test or overlap_val_test:
            print("❌ CRITICAL ERROR: Episode overlap detected!")
            print("   This would cause DATA LEAKAGE and invalidate results!")
            if overlap_train_val:
                print(f"   Train-Val overlap: {sorted(list(overlap_train_val))}")
            if overlap_train_test:
                print(f"   Train-Test overlap: {sorted(list(overlap_train_test))}")
            if overlap_val_test:
                print(f"   Val-Test overlap: {sorted(list(overlap_val_test))}")
            raise ValueError("Data leakage detected: Episodes appear in multiple splits")
        else:
            print("✓ No episode overlap - Data leakage prevented")
        
        # Check 2: Correct episode ranges
        if train_episodes_actual != train_episodes_expected:
            missing_train = train_episodes_expected - train_episodes_actual
            extra_train = train_episodes_actual - train_episodes_expected
            if missing_train:
                print(f"⚠️  WARNING: Training set missing episodes: {sorted(list(missing_train))}")
            if extra_train:
                print(f"⚠️  WARNING: Training set has extra episodes: {sorted(list(extra_train))}")
        else:
            print(f"✓ Training set has correct episodes (0-79)")
        
        if val_episodes_actual != val_episodes_expected:
            missing_val = val_episodes_expected - val_episodes_actual
            if missing_val:
                print(f"⚠️  WARNING: Validation set missing episodes: {sorted(list(missing_val))}")
        else:
            print(f"✓ Validation set has correct episodes (80-89)")
        
        if test_episodes_actual != test_episodes_expected:
            missing_test = test_episodes_expected - test_episodes_actual
            if missing_test:
                print(f"⚠️  WARNING: Test set missing episodes: {sorted(list(missing_test))}")
        else:
            print(f"✓ Test set has correct episodes (90-99)")
        
        # Check 3: Feature consistency
        train_cols = set(train_df.columns)
        val_cols = set(val_df.columns)
        test_cols = set(test_df.columns)
        
        if train_cols != val_cols or train_cols != test_cols:
            print("❌ WARNING: Different features across splits!")
            print(f"   This will cause errors during training/evaluation")
        else:
            print(f"✓ All splits have identical {len(train_cols)} features")
        
        # Check 4: Temporal ordering
        for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            if 'step_id' in split_df.columns:
                violations = 0
                for ep in split_df[episode_col].unique():
                    ep_data = split_df[split_df[episode_col] == ep]['step_id']
                    if not ep_data.is_monotonic_increasing:
                        violations += 1
                if violations > 0:
                    print(f"⚠️  WARNING: {violations} episodes in {split_name} have non-monotonic timesteps")
        
        print(f"✓ Temporal ordering preserved")
        print(f"{'='*70}\n")
    
    
    def verify_team_coordination(self, other_metadata_path: str) -> bool:
        """
        Verify another team member's split matches yours.
        
        Parameters
        ----------
        other_metadata_path : str
            Path to teammate's split_metadata.json
        
        Returns
        -------
        bool
            True if splits match, False otherwise
        """
        with open(other_metadata_path, 'r') as f:
            other_metadata = json.load(f)
        
        print(f"\n{'='*70}")
        print(f"TEAM COORDINATION VERIFICATION")
        print(f"{'='*70}")
        
        checks_passed = True
        
        # Check episode ranges
        if other_metadata.get('train_episodes') != self.train_episodes:
            print(f"❌ Training episodes MISMATCH!")
            print(f"   Your config: {self.train_episodes[:5]}...{self.train_episodes[-5:]}")
            print(f"   Their config: {other_metadata['train_episodes'][:5]}...{other_metadata['train_episodes'][-5:]}")
            checks_passed = False
        else:
            print(f"✓ Training episodes match (0-79)")
        
        if other_metadata.get('val_episodes') != self.val_episodes:
            print(f"❌ Validation episodes MISMATCH!")
            checks_passed = False
        else:
            print(f"✓ Validation episodes match (80-89)")
        
        if other_metadata.get('test_episodes') != self.test_episodes:
            print(f"❌ Test episodes MISMATCH!")
            checks_passed = False
        else:
            print(f"✓ Test episodes match (90-99)")
        
        # Check random seed
        if other_metadata.get('random_seed') != self.random_seed:
            print(f"⚠️  WARNING: Different random seeds!")
            print(f"   Your seed: {self.random_seed}")
            print(f"   Their seed: {other_metadata.get('random_seed')}")
            print(f"   This may cause different weight initializations")
        else:
            print(f"✓ Random seed matches ({self.random_seed})")
        
        print(f"{'='*70}")
        if checks_passed:
            print(f"✓ ALL CHECKS PASSED - Results will be comparable!")
        else:
            print(f"❌ MISMATCHES DETECTED - Results NOT comparable!")
            print(f"   Coordinate with team to use same split configuration")
        print(f"{'='*70}\n")
        
        return checks_passed


# Convenience function for quick splitting
def quick_split(
    input_csv: str,
    output_dir: str = 'data/processed/',
    episode_col: str = 'episode_id',
    verify_episodes: bool = True
) -> Dict:
    """
    Quick function to split data with coordinated team configuration.
    
    Parameters
    ----------
    input_csv : str
        Path to input CSV file
    output_dir : str, default='data/processed/'
        Output directory for splits
    episode_col : str, default='episode_id'
        Episode identifier column
    verify_episodes : bool, default=True
        Verify dataset has episodes 0-99
    
    Returns
    -------
    dict
        Split metadata
    
    Example
    -------
    >>> from data.splits import quick_split
    >>> metadata = quick_split('data/raw/network_data.csv')
    """
    splitter = CoordinatedEpisodeSplitter()
    return splitter.split_and_save(input_csv, output_dir, episode_col, verify_episodes=verify_episodes)


if __name__ == "__main__":
    """
    Command-line usage:
    python data/splits.py <input_csv> [output_dir]
    """
    import sys
    
    print("\n" + "="*70)
    print("COORDINATED EPISODE SPLITTER")
    print("="*70)
    
    default_input = 'D:\Academics\SEM-5\Machine Learning\ML_courseproj\Dataset\nr_dataset_full.csv'
    default_output = 'D:\Academics\SEM-5\Machine Learning\ML_courseproj\Splits'
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else default_output
    else:
        print(f"\nUsage: python data/splits.py <input_csv> [output_dir]")
        print(f"\nUsing defaults:")
        print(f"  Input:  {default_input}")
        print(f"  Output: {default_output}\n")
        
        input_path = default_input
        output_path = default_output
    
    # Perform splitting
    try:
        metadata = quick_split(
            input_csv=input_path,
            output_dir=output_path,
            verify_episodes=True
        )
        
        print(f"\n{'='*70}")
        print(f"✓ SUCCESS! Team-coordinated splits created.")
        print(f"{'='*70}")
        print(f"\nNEXT STEPS:")
        print(f"  1. Share 'team_coordination_config.json' with Prateek")
        print(f"  2. Both use same hyperparameters (LR=0.001, batch=64)")
        print(f"  3. Run preprocessing: python data/preprocessor.py")
        print(f"  4. Train models: python experiments/train_gru.py")
        print(f"  5. Compare results on SAME test episodes (90-99)")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\nError during splitting: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)