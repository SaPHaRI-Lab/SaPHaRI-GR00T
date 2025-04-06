import pandas as pd
import numpy as np
import glob, json, os

parquet_folder = "data"            # Folder containing .parquet files
output_path = "meta/stats.json"    # Output file

FIELDS = [
    "observation.state",
    "action",
    "timestamp",
    "task_index",
    "annotation.human.action.task_description",
    "annotation.human.validity",
    "episode_index",
    "index",
    "next.reward",
    "next.done"
]

data_accumulator = {key: [] for key in FIELDS}

for file in glob.glob(os.path.join(parquet_folder, "*.parquet")):
    df = pd.read_parquet(file)
    for key in FIELDS:
        if key in df.columns:
            values = df[key].apply(lambda x: x if isinstance(x, list) else [x])
            stacked = stacked = np.vstack(values.tolist())
            data_accumulator[key].append(stacked)

def compute_stats(arr):
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "q01": np.percentile(arr, 1, axis=0).tolist(),
        "q99": np.percentile(arr, 99, axis=0).tolist()
    }

stats = {}
for key, collected in data_accumulator.items():
    if collected:
        combined = np.vstack(collected)
        stats[key] = compute_stats(combined)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(stats, f, indent=2)

print(f"✅ Full stats.json saved to: {output_path}")
