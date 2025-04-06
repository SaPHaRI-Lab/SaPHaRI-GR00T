import pandas as pd
import numpy as np
import json
import os

# CONFIG
input_parquet = "episode_000000.parquet"
output_json = "episode_000000_stats.json"

# Load parquet
df = pd.read_parquet(input_parquet)

# Init storage
stats = {}

# Compute function
def compute_stats(arr):
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "q01": np.percentile(arr, 1, axis=0).tolist(),
        "q99": np.percentile(arr, 99, axis=0).tolist()
    }

#
for col in df.columns:
    print(f"Processing: {col}")
    try:
        # Ensure list type (e.g., [14-dim] or scalar wrapped as [x])
        values = df[col].apply(lambda x: x if isinstance(x, list) else [x])
        data = np.vstack(values.tolist())
        stats[col] = compute_stats(data)
    except Exception as e:
        print(f"⚠️ Skipped {col}: {e}")

#JSON
os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, "w") as f:
    json.dump(stats, f, indent=2)

print(f"stats.json saved to: {output_json}")
