import pandas as pd
import numpy as np
import os.path as path
from pathlib import Path
import os, argparse, json, shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', help='The dataset folder', default='novideo_data')
    parser.add_argument('--reduced', '-r', action='store_true', help='If the reduced versions of the CSVs should be used instead')
    parser.add_argument('--fps', help='If the reduced versions of the CSVs should be used instead', default=20)
    args = parser.parse_args()
    # CONFIG
    base_folder = Path(args.folder)
    if not base_folder.exists():
        raise Exception("No such folder exists")
    data_folder = base_folder / 'raw_data_files'

    # Grab all of the csvs in the given directory and sort by alphabetical order
    csvs = [file for file in (data_folder / 'reduced_csvs').glob('*.csv')] if args.reduced else [file for file in data_folder.glob('*.csv')]
    csvs = sorted(csvs)
    # Load the json that holds the names of all the gestures and their description
    description = json.load(open(base_folder / "prompts.json"))    
    assert len(description) == len(csvs), "Number of gesture descriptions don't match the number of csvs"
    
    print("Sorted Order for CSVS & Description:")
    for i, csv in enumerate(csvs):
        # Add the index of the gesture in 'csvs' that corresponds the entry in the dictionary 
        description[csv.stem] = (description[csv.stem], i)
        print(f'\t {csv.name} - : {description[csv.stem][0]}, {description[csv.stem][1]}')
    print('\n')
    fps = args.fps
    # TODO: Check what these numbers should be in the dataset frame based? ID based? Are they not used?
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
    steps_per_task = {}
    for i, csv in enumerate(csvs):
        df = pd.read_csv(csv)
        if all([inp_col[-2:] == exp_col[-2:] for inp_col, exp_col in zip(df.columns, input_cols)]):
            # df = df[desired_order]
            print("Columns were out of order. Sorting columns")

        reordered_df = df[desired_order]
        data = reordered_df.to_numpy()
        num_frames = data.shape[0]
        # Build output DataFrame
        out = pd.DataFrame()
        out["observation.state"] = reordered_df.values.tolist()
        # Must be a list to added as a column
        out["action"] = np.vstack([data[1:], [data[-1]]]).tolist()
        out["timestamp"] = (np.arange(num_frames) / fps).round(5)
        out["annotation.human.action.task_description"] = [description[csv.stem]] * num_frames

        out["task_index"] = [description[csv.stem]] * num_frames # index of the task description in the meta/tasks.jsonl file
        out["annotation.human.validity"] = [len(csvs)] * num_frames # index of the task in the meta/tasks.jsonl file
        out["episode_index"] = [description[csv.stem]] * num_frames # index of the episode
        out["index"] = list(range(num_frames))
        out["next.reward"] = [0.0] * num_frames
        out["next.done"] = [False] * num_frames

        # Save to parquet
        out.to_parquet(base_folder / 'data' / 'chunk-000' / f"episode_{i:06d}.parquet", index=False)
        print(f"✅ Saved \033[91m{csv}\033[0m to: '\033[93m'episode_{i:06d}.parquet\033[0m")
        # Save the number of steps in the gesture for episodes.jsonl
        steps_per_task[csv.stem] = len(df)
    # TODO: Flag if the number of steps don't match the number of frames in the video
    # TODO: Account for extra frames in each video and mapping from video to gesture
    # Save and rename videos into the designated folder
    src = data_folder / 'Videos'
    for video in src.glob("*.mp4"):
        dst = base_folder / 'videos' / 'chunk-000' / 'observation.images.ego_view' / f"episode_{description[video.stem][1]:06d}.mp4"
        if dst.exists():
            shutil.copy(video, dst)
            print(f"Copied \033[91m{video}\033[0m to \033[91m{dst}\033[0m")
        else:
            print("Destination folder does not exist:", dst.relative_to(data_folder))
    # Update episodes.jsonl with the new gestures
    with open('novideo_data/meta/episodes.jsonl', 'w') as file:
        for i, csv in enumerate(csvs):
            line = {"episode_index": description[csv.stem][1], "tasks": [description[csv.stem][0], "valid"], "length": steps_per_task[csv.stem]}
            json.dump(line, file)
            file.write("\n")
    with open('novideo_data/meta/tasks.jsonl', 'w') as file:
        for i, csv in enumerate(csvs):
            line = {"task_index": description[csv.stem][1], "task": description[csv.stem][0]}
            json.dump(line, file)
            file.write("\n")
        json.dump({"task_index": len(csvs), "task": "valid"}, file)