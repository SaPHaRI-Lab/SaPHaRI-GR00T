import pandas as pd
import numpy as np

# CONFIG
input_csv = "novideo_data/CSV_files/sup.csv"
output_parquet = "episode_000003.parquet"
fps = 20.0
task_id = 0
episode_index = 0

# Expected input column order (your format)
input_cols = [
    "left_w0", "left_w1", "left_w2",
    "left_e0", "left_e1",
    "left_s0", "left_s1",
    "right_s0", "right_s1",
    "right_w0", "right_w1", "right_w2",
    "right_e0", "right_e1"
]

# Desired output order for LeRobot (based on GR00T modality.json)
desired_order = [
    "left_s0", "left_s1", "left_e0", "left_e1",
    "left_w0", "left_w1", "left_w2",
    "right_s0", "right_s1", "right_e0", "right_e1",
    "right_w0", "right_w1", "right_w2"
]

# Load and reorder
df = pd.read_csv(input_csv)
if list(df.columns) != input_cols:
    raise ValueError("CSV does not match expected input column order.")

reordered_df = df[desired_order]
data = reordered_df.to_numpy()
num_frames = data.shape[0]

# Build output DataFrame
out = pd.DataFrame()
out["observation.state"] = reordered_df.values.tolist()
out["action"] = np.vstack(([np.zeros_like(data[0])], np.diff(data, axis=0))).tolist()
out["timestamp"] = (np.arange(num_frames) / fps).round(5)
out["annotation.human.action.task_description"] = [task_id] * num_frames
out["task_index"] = [task_id] * num_frames
out["annotation.human.validity"] = [1] * num_frames
out["episode_index"] = [episode_index] * num_frames
out["index"] = list(range(num_frames))
out["next.reward"] = [0.0] * num_frames
out["next.done"] = [False] * num_frames

# Save to parquet
out.to_parquet(output_parquet, index=False)
print(f"✅ Saved to: {output_parquet}")
