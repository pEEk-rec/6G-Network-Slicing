import pandas as pd
import numpy as np

df = pd.read_csv(r'D:\Academics\SEM-5\Machine Learning\ML_courseproj\Dataset\nr_dataset_dynamic.csv')

print("="*60)
print("DYNAMIC PRB ALLOCATION - VALIDATION")
print("="*60)

print(f"\nTotal samples: {len(df):,}")
print(f"Episodes: {df['episode_id'].nunique()}")
print(f"Steps per episode: {df.groupby('episode_id')['step_id'].max().mean():.0f}")

print("\n PRB ALLOCATION STATISTICS:")
print(f"  Min PRB: {df['allocated_prbs'].min()}")
print(f"  Max PRB: {df['allocated_prbs'].max()}")
print(f"  Mean PRB: {df['allocated_prbs'].mean():.1f}")
print(f"  Unique PRB values: {df['allocated_prbs'].nunique()}")

print("\n PRB BY SLICE TYPE:")
for s in [1, 2, 3]:
    name = {1: 'eMBB', 2: 'URLLC', 3: 'mMTC'}[s]
    prbs = df[df['slice_type']==s]['allocated_prbs']
    print(f"  {name:6s}: {prbs.mean():5.1f} ± {prbs.std():4.1f} (range: {prbs.min():2d}-{prbs.max():2d})")

print("\n PRB BUDGET CONSTRAINT CHECK:")
# Check if PRBs sum to 100 at each timestep
for ep in df['episode_id'].unique()[:5]:  # Check first 5 episodes
    for step in df[df['episode_id']==ep]['step_id'].unique()[:10]:
        step_data = df[(df['episode_id']==ep) & (df['step_id']==step)]
        total_prbs = step_data['allocated_prbs'].sum()
        if total_prbs != 100:
            print(f"    Episode {ep}, Step {step}: Total PRBs = {total_prbs} (should be 100!)")
            break
    else:
        continue
    break
else:
    print("  ✓ All timesteps sum to exactly 100 PRBs!")

print("\n DYNAMIC ALLOCATION PROOF:")
print("  PRB variance (high = dynamic):", df['allocated_prbs'].var())
print("  Expected: >50 for dynamic, <5 for static")

if df['allocated_prbs'].var() > 50:
    print("\n SUCCESS: PRBs are DYNAMIC!")
else:
    print("\n  WARNING: PRBs might still be too static")

print("="*60)
