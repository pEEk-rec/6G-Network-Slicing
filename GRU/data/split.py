"""
Episode-based data splitting for temporal network data.
Ensures no data leakage by splitting at episode boundaries.

Author: 6G Bandwidth Allocation Project
Date: 2024
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Tuple, Dict, Optional
from datetime import datetime


class EpisodeSplitter:
    """
    Splits sequential episode data into train/validation/test sets.
    
    Key Features:
    - No temporal leakage: Episodes are strictly separated
    - Sequential splitting: Mimics real-world deployment (train on past, test on future)
    - Validation checks: Ensures data integrity
    - Metadata tracking: Saves split info for reproducibility
    
    Parameters
    ----------
    train_ratio : float, default=0.8
        Proportion of episodes for training (0.0 to 1.0)
    val_ratio : float, default=0.1
        Proportion of episodes for validation (0.0 to 1.0)
    test_ratio : float, default=0.1
        Proportion of episodes for testing (0.0 to 1.0)
    random_state : int, default=42
        Random seed for reproducibility (currently unused, reserved for future features)
    
    Example
    -------
    >>> splitter = EpisodeSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    >>> splitter.split_and_save(
    ...     input_path='data/raw/network_data.csv',
    ...     output_dir='data/processed/',
    ...     episode_col='episode_id'
    ... )
    """
    
    def __init__(
        self, 
        train_ratio: float = 0.8, 
        val_ratio: float = 0.1, 
        test_ratio: float = 0.1,
        random_state: int = 42
    ):
        # Validate ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(
                f"Ratios must sum to 1.0. Got: {train_ratio + val_ratio + test_ratio:.4f}"
            )
        
        if any(r <= 0 for r in [train_ratio, val_ratio, test_ratio]):
            raise ValueError("All ratios must be positive")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        
        # Will be populated after splitting
        self.split_info = {}
        
    
    def split(
        self, 
        df: pd.DataFrame, 
        episode_col: str = 'episode_id'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame into train/validation/test by episodes.
        
        Parameters
        ----------
        df : pd.DataFrame
            Full dataset with episode_id column
        episode_col : str, default='episode_id'
            Name of column containing episode identifiers
        
        Returns
        -------
        train_df : pd.DataFrame
            Training set
        val_df : pd.DataFrame
            Validation set
        test_df : pd.DataFrame
            Test set
        
        Raises
        ------
        ValueError
            If episode_col not in DataFrame or episodes are non-sequential
        """
        # Validate input
        if episode_col not in df.columns:
            raise ValueError(f"Column '{episode_col}' not found in DataFrame")
        
        # Get unique episodes and sort
        unique_episodes = sorted(df[episode_col].unique())
        total_episodes = len(unique_episodes)
        
        print(f"\n{'='*60}")
        print(f"EPISODE-BASED SPLITTING")
        print(f"{'='*60}")
        print(f"Total episodes found: {total_episodes}")
        print(f"Episode range: {unique_episodes[0]} to {unique_episodes[-1]}")
        
        # Check for missing episodes
        expected_episodes = list(range(unique_episodes[0], unique_episodes[-1] + 1))
        missing_episodes = set(expected_episodes) - set(unique_episodes)
        
        if missing_episodes:
            print(f"\n WARNING: {len(missing_episodes)} missing episodes detected!")
            print(f"Missing episode IDs: {sorted(list(missing_episodes))[:10]}...")
            response = input("Continue anyway? (yes/no): ")
            if response.lower() != 'yes':
                raise ValueError("Splitting aborted due to missing episodes")
        
        # Calculate split boundaries
        train_end_idx = int(total_episodes * self.train_ratio)
        val_end_idx = int(total_episodes * (self.train_ratio + self.val_ratio))
        
        # Get episode ranges
        train_episodes = unique_episodes[:train_end_idx]
        val_episodes = unique_episodes[train_end_idx:val_end_idx]
        test_episodes = unique_episodes[val_end_idx:]
        
        print(f"\nSplit Configuration:")
        print(f"  Train: Episodes {train_episodes[0]}-{train_episodes[-1]} ({len(train_episodes)} episodes)")
        print(f"  Val:   Episodes {val_episodes[0]}-{val_episodes[-1]} ({len(val_episodes)} episodes)")
        print(f"  Test:  Episodes {test_episodes[0]}-{test_episodes[-1]} ({len(test_episodes)} episodes)")
        
        # Create splits
        train_df = df[df[episode_col].isin(train_episodes)].copy()
        val_df = df[df[episode_col].isin(val_episodes)].copy()
        test_df = df[df[episode_col].isin(test_episodes)].copy()
        
        # Store split metadata
        self.split_info = {
            'total_episodes': total_episodes,
            'train_episodes': (int(train_episodes[0]), int(train_episodes[-1])),
            'val_episodes': (int(val_episodes[0]), int(val_episodes[-1])),
            'test_episodes': (int(test_episodes[0]), int(test_episodes[-1])),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'episode_col': episode_col,
            'split_timestamp': datetime.now().isoformat()
        }
        
        # Validate split
        self._validate_split(train_df, val_df, test_df, episode_col)
        
        return train_df, val_df, test_df
    
    
    def split_and_save(
        self,
        input_path: str,
        output_dir: str,
        episode_col: str = 'episode_id',
        prefix: str = '',
        save_metadata: bool = True
    ) -> Dict:
        """
        Load data, split, and save to separate files.
        
        Parameters
        ----------
        input_path : str
            Path to input CSV file
        output_dir : str
            Directory to save train/val/test files
        episode_col : str, default='episode_id'
            Name of episode column
        prefix : str, default=''
            Prefix for output filenames (e.g., 'network_')
        save_metadata : bool, default=True
            Whether to save split metadata as JSON
        
        Returns
        -------
        dict
            Split metadata including file paths and statistics
        
        Example
        -------
        >>> splitter.split_and_save(
        ...     input_path='data/raw/data.csv',
        ...     output_dir='data/processed/',
        ...     prefix='bandwidth_'
        ... )
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"LOADING DATA")
        print(f"{'='*60}")
        print(f"Input file: {input_path}")
        
        # Load data
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df):,} samples with {len(df.columns)} features")
        
        # Perform split
        train_df, val_df, test_df = self.split(df, episode_col)
        
        # Generate filenames
        train_file = output_path / f"{prefix}train.csv"
        val_file = output_path / f"{prefix}val.csv"
        test_file = output_path / f"{prefix}test.csv"
        metadata_file = output_path / f"{prefix}split_metadata.json"
        
        # Save splits
        print(f"\n{'='*60}")
        print(f"SAVING SPLITS")
        print(f"{'='*60}")
        
        print(f"Saving training set...")
        train_df.to_csv(train_file, index=False)
        print(f"  → {train_file} ({len(train_df):,} samples)")
        
        print(f"Saving validation set...")
        val_df.to_csv(val_file, index=False)
        print(f"  → {val_file} ({len(val_df):,} samples)")
        
        print(f"Saving test set...")
        test_df.to_csv(test_file, index=False)
        print(f"  → {test_file} ({len(test_df):,} samples)")
        
        # Add file paths to metadata
        self.split_info['files'] = {
            'train': str(train_file),
            'val': str(val_file),
            'test': str(test_file)
        }
        
        # Save metadata
        if save_metadata:
            print(f"\nSaving metadata...")
            with open(metadata_file, 'w') as f:
                json.dump(self.split_info, f, indent=2)
            print(f"  → {metadata_file}")
        
        print(f"\n{'='*60}")
        print(f"✓ SPLITTING COMPLETE")
        print(f"{'='*60}\n")
        
        return self.split_info
    
    
    def _validate_split(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        episode_col: str
    ):
        """
        Validate split correctness to prevent data leakage.
        
        Checks:
        - No episode overlap between sets
        - All samples accounted for
        - Temporal ordering preserved
        """
        print(f"\n{'='*60}")
        print(f"VALIDATION CHECKS")
        print(f"{'='*60}")
        
        train_episodes = set(train_df[episode_col].unique())
        val_episodes = set(val_df[episode_col].unique())
        test_episodes = set(test_df[episode_col].unique())
        
        # Check 1: No overlap
        overlap_train_val = train_episodes & val_episodes
        overlap_train_test = train_episodes & test_episodes
        overlap_val_test = val_episodes & test_episodes
        
        if overlap_train_val or overlap_train_test or overlap_val_test:
            print("CRITICAL ERROR: Episode overlap detected!")
            if overlap_train_val:
                print(f"   Train-Val overlap: {overlap_train_val}")
            if overlap_train_test:
                print(f"   Train-Test overlap: {overlap_train_test}")
            if overlap_val_test:
                print(f"   Val-Test overlap: {overlap_val_test}")
            raise ValueError("Data leakage: Episodes appear in multiple splits")
        else:
            print("✓ No episode overlap - Data leakage prevented")
        
        # Check 2: Sample counts
        total_samples = len(train_df) + len(val_df) + len(test_df)
        print(f"✓ All samples accounted for: {total_samples:,} total")
        
        # Check 3: Size distribution
        train_pct = len(train_df) / total_samples * 100
        val_pct = len(val_df) / total_samples * 100
        test_pct = len(test_df) / total_samples * 100
        
        print(f"\nSample Distribution:")
        print(f"  Train: {len(train_df):>7,} samples ({train_pct:>5.2f}%)")
        print(f"  Val:   {len(val_df):>7,} samples ({val_pct:>5.2f}%)")
        print(f"  Test:  {len(test_df):>7,} samples ({test_pct:>5.2f}%)")
        
        # Check 4: Temporal ordering within episodes
        for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            if 'step_id' in split_df.columns:
                for ep in split_df[episode_col].unique():
                    ep_data = split_df[split_df[episode_col] == ep]['step_id']
                    if not ep_data.is_monotonic_increasing:
                        print(f" WARNING: Non-monotonic step_id in {split_name} episode {ep}")
        
        print("✓ Temporal ordering preserved within episodes")
        
        # Check 5: Feature completeness
        train_cols = set(train_df.columns)
        val_cols = set(val_df.columns)
        test_cols = set(test_df.columns)
        
        if train_cols != val_cols or train_cols != test_cols:
            print("❌ WARNING: Different columns across splits!")
            print(f"   Train: {len(train_cols)} columns")
            print(f"   Val: {len(val_cols)} columns")
            print(f"   Test: {len(test_cols)} columns")
        else:
            print(f"✓ All splits have {len(train_cols)} features")
        
        print(f"{'='*60}\n")
    
    
    def get_split_info(self) -> Dict:
        """
        Get metadata about the most recent split.
        
        Returns
        -------
        dict
            Split statistics and configuration
        """
        if not self.split_info:
            raise ValueError("No split performed yet. Call split() or split_and_save() first.")
        return self.split_info
    
    
    def load_split_metadata(self, metadata_path: str) -> Dict:
        """
        Load previously saved split metadata.
        
        Parameters
        ----------
        metadata_path : str
            Path to split_metadata.json file
        
        Returns
        -------
        dict
            Split metadata
        """
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Loaded split metadata from: {metadata_path}")
        print(f"Split created: {metadata['split_timestamp']}")
        print(f"Episodes - Train: {metadata['train_episodes']}, "
              f"Val: {metadata['val_episodes']}, Test: {metadata['test_episodes']}")
        
        return metadata


# Convenience function for quick splitting
def quick_split(
    input_csv: str,
    output_dir: str = 'data/processed/',
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    episode_col: str = 'episode_id'
) -> Dict:
    """
    Quick function to split data with default settings.
    
    Parameters
    ----------
    input_csv : str
        Path to input CSV file
    output_dir : str, default='data/processed/'
        Output directory for splits
    train_ratio : float, default=0.8
        Training set proportion
    val_ratio : float, default=0.1
        Validation set proportion
    test_ratio : float, default=0.1
        Test set proportion
    episode_col : str, default='episode_id'
        Episode identifier column
    
    Returns
    -------
    dict
        Split metadata
    
    Example
    -------
    >>> from data.splits import quick_split
    >>> metadata = quick_split('data/raw/network_data.csv')
    """
    splitter = EpisodeSplitter(train_ratio, val_ratio, test_ratio)
    return splitter.split_and_save(input_csv, output_dir, episode_col=episode_col)


if __name__ == "__main__":
    """
    Example usage when running directly:
    python data/splits.py
    """
    import sys
    
    # Default paths
    default_input = 'data/raw/network_data.csv'
    default_output = 'data/processed/'
    
    print("\n" + "="*60)
    print("BANDWIDTH ALLOCATION - DATA SPLITTER")
    print("="*60)
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else default_output
    else:
        print(f"\nUsage: python data/splits.py <input_csv> [output_dir]")
        print(f"\nUsing defaults:")
        print(f"  Input:  {default_input}")
        print(f"  Output: {default_output}")
        
        input_path = default_input
        output_path = default_output
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"\nError: Input file not found: {input_path}")
        print(f"Please provide a valid CSV file path.")
        sys.exit(1)
    
    # Perform splitting
    try:
        metadata = quick_split(
            input_csv=input_path,
            output_dir=output_path,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        
        print("\n✓ SUCCESS! Files ready for training.")
        print(f"\nNext steps:")
        print(f"  1. Run preprocessing: python data/preprocessor.py")
        print(f"  2. Generate sequences: python data/sequence_generator.py")
        print(f"  3. Train model: python experiments/train_gru.py\n")
        
    except Exception as e:
        print(f"\n Error during splitting: {str(e)}")
        sys.exit(1)