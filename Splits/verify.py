import pandas as pd

# Load splits
train = pd.read_csv(r'D:\Academics\SEM-5\Machine Learning\ML_courseproj\Splits\train.csv')
val = pd.read_csv(r'D:\Academics\SEM-5\Machine Learning\ML_courseproj\Splits\val.csv')
test = pd.read_csv(r'D:\Academics\SEM-5\Machine Learning\ML_courseproj\Splits\test.csv')

print("=== SPLIT VERIFICATION ===")
print(f"\nTrain: {len(train):,} samples")
print(f"  Episodes: {train['episode_id'].min()} to {train['episode_id'].max()}")
print(f"  Unique episodes: {train['episode_id'].nunique()}")

print(f"\nVal: {len(val):,} samples")
print(f"  Episodes: {val['episode_id'].min()} to {val['episode_id'].max()}")
print(f"  Unique episodes: {val['episode_id'].nunique()}")

print(f"\nTest: {len(test):,} samples")
print(f"  Episodes: {test['episode_id'].min()} to {test['episode_id'].max()}")
print(f"  Unique episodes: {test['episode_id'].nunique()}")

# Check for overlap
train_eps = set(train['episode_id'].unique())
val_eps = set(val['episode_id'].unique())
test_eps = set(test['episode_id'].unique())

assert len(train_eps & val_eps) == 0, "Train-Val overlap!"
assert len(train_eps & test_eps) == 0, "Train-Test overlap!"
assert len(val_eps & test_eps) == 0, "Val-Test overlap!"
print("\n✓ No data leakage - All splits are clean!")

# Check slice distribution
print("\n=== SLICE DISTRIBUTION ===")
for split_name, df in [('Train', train), ('Val', val), ('Test', test)]:
    print(f"\n{split_name}:")
    slice_counts = df['slice_type'].value_counts().sort_index()
    for slice_id, count in slice_counts.items():
        slice_name = {1: 'eMBB', 2: 'URLLC', 3: 'mMTC'}[slice_id]
        print(f"  {slice_name}: {count:,} samples ({count/len(df)*100:.1f}%)")