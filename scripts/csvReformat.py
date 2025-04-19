import pandas as pd
import numpy as np
import os.path as path
import os, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', help='The folder with each CSV file', default='novideo_data/CSV_files/')
    parser.add_argument('-t', '--task_order', 
            help='The episode number to give to each csv, in alphabetical order (order of the files)', 
            default=[1, 2, 3, 0], type=list
    )
    args = parser.parse_args()
    # CONFIG
    csvs = [os.path.join(args.folder, file) for file in os.listdir(args.folder) if os.path.isfile(os.path.join(args.folder, file))]
    print(csvs)
    output_parquet = lambda i: f"episode_00000{i}.parquet"
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
    for i, csv in enumerate(csvs):
        df = pd.read_csv(csv)
        if all([inp_col[-2:] == exp_col[-2:] for inp_col, exp_col in zip(df.columns, input_cols)]):
            df = df[desired_order]
            print("Columns were out of order. Sorting columns")

        reordered_df = df[desired_order]
        data = reordered_df.to_numpy()
        num_frames = data.shape[0]
        # Build output DataFrame
        out = pd.DataFrame()
        print(data[1:].shape, data[-1:].shape)
        out["observation.state"] = reordered_df.values.tolist()
        # Must be a list to added as a column
        out["action"] = np.vstack([data[1:], [data[-1]]]).tolist()
        out["timestamp"] = (np.arange(num_frames) / fps).round(5)
        out["annotation.human.action.task_description"] = [task_id] * num_frames
        out["task_index"] = [task_id] * num_frames

        out["annotation.human.validity"] = [1] * num_frames
        out["episode_index"] = [episode_index] * num_frames
        out["index"] = list(range(num_frames))
        out["next.reward"] = [0.0] * num_frames
        out["next.done"] = [False] * num_frames

        # Save to parquet
        
        out.to_parquet(os.path.join('novideo_data/data/chunk-000', output_parquet(args.task_order[i])), index=False)
        print(f"✅ Saved {csv} to: {output_parquet(args.task_order[i])}")


