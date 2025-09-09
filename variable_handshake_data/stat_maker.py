import pandas as pd
import numpy as np
import json
import os
import glob

input_folder = "novideo_data/CSV_files"
output_json = "/Users/anchen/Documents/GitHub/SaPHaRI-GR00T/novideo_data/jsonfile_novideo/stats.json"
fps = 20.0

stats = {}
accumulator = {
    "observation.state": [],
    "action": [],
    "timestamp": [],
    "task_index": [],
    "annotation.human.action.task_description": [],
    "annotation.human.validity": [],
    "episode_index": [],
    "index": []
}

def compute_stats(arr):
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "q01": np.percentile(arr, 1, axis=0).tolist(),
        "q99": np.percentile(arr, 99, axis=0).tolist()
    }

csv_files = sorted(glob.glob(os.path.join(input_folder, "*.csv")))
frame_counter = 0

for episode_idx, file in enumerate(csv_files):
    try:
        df = pd.read_csv(file, skiprows=1, header=None)
        data = df.to_numpy()

        deltas = np.diff(data, axis=0)
        num_frames = len(data)

        accumulator["observation.state"].append(data)
        accumulator["action"].append(deltas)

        timestamps = np.arange(num_frames) / fps
        accumulator["timestamp"].append(timestamps[:, None])

        task_id = episode_idx  # or use a filename-to-id map
        task_ids = np.full((num_frames, 1), task_id)
        accumulator["task_index"].append(task_ids)
        accumulator["annotation.human.action.task_description"].append(task_ids)

        valid = np.ones((num_frames, 1))
        accumulator["annotation.human.validity"].append(valid)

        episode_ids = np.full((num_frames, 1), episode_idx)
        accumulator["episode_index"].append(episode_ids)

        indices = np.arange(frame_counter, frame_counter + num_frames).reshape(-1, 1)
        accumulator["index"].append(indices)

        frame_counter += num_frames

    except Exception as e:
        print(f"Skipped {file}: {e}")

for field, all_data in accumulator.items():
    combined = np.vstack(all_data)
    stats[field] = compute_stats(combined)

os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, "w") as f:
    json.dump(stats, f, indent=2)

print(f"stats.json saved to: {output_json}")
