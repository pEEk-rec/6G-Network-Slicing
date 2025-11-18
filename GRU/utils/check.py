import pandas as pd
import numpy as np

# Load one processed CSV (before sequence generation)
df = pd.read_csv(r"D:\Academics\SEM-5\Machine Learning\ML_courseproj\Splits\train_dataset.csv")

# Calculate correlations
features = df[['embb_demand', 'urllc_demand', 'mmtc_demand', 
               'sinr_db_embb', 'sinr_db_urllc', 'sinr_db_mmtc',
               'embb_utilization', 'urllc_utilization', 'mmtc_utilization']]
targets = df[['allocated_bandwidth_mbps_embb', 'allocated_bandwidth_mbps_urllc', 'allocated_bandwidth_mbps_mmtc']]

print("Feature-Target Correlations:")
print(pd.concat([features, targets], axis=1).corr().loc[features.columns, targets.columns])
