import warnings
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy, BasePolicy
from gr00t.data.schema import EmbodimentTag
from gr00t.data.dataset import LeRobotSingleDataset
import numpy as np
from tqdm import tqdm
import torch
import argparse, matplotlib.pyplot as plt
import json, os, pandas as pd
PRE_TRAINED_MODEL_PATH = "nvidia/GR00T-N1-2B"
DATASET_PATH = "novideo_data/"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def predict_actions(
    policy: BasePolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    modality_keys: list,
    steps=300,
    action_horizon=16,
    policy_name="Fine-tuned"
):
    state_joints_across_time = []
    gt_action_joints_across_time = []
    pred_action_joints_across_time = []
    
    for step_count in tqdm(range(steps)):
        data_point = dataset.get_step_data(traj_id, step_count)

        # NOTE this is to get all modality keys concatenated
        # concat_state = data_point[f"state.{modality_keys[0]}"][0]
        # concat_gt_action = data_point[f"action.{modality_keys[0]}"][0]
        concat_state = np.concatenate(
            [data_point[f"state.{key}"][0] for key in modality_keys], axis=0
        )
        concat_gt_action = np.concatenate(
            [data_point[f"action.{key}"][0] for key in modality_keys], axis=0
        )

        state_joints_across_time.append(concat_state)
        gt_action_joints_across_time.append(concat_gt_action)

        if step_count % action_horizon == 0:
            print("inferencing at step: ", step_count)
            action_chunk = policy.get_action(data_point)
            for j in range(action_horizon):
                # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                concat_pred_action = np.concatenate(
                    [np.atleast_1d(action_chunk[f"action.{key}"][j]) for key in modality_keys],
                    axis=0,
                )
                pred_action_joints_across_time.append(concat_pred_action)
    return state_joints_across_time, gt_action_joints_across_time, pred_action_joints_across_time

def read_jsonl(file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    json_object = json.loads(line)
                    data.append(json_object)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line.strip()}")
        return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-em', '--embodiment_tag', 
        help='The embodiment tag that is used to finetune the dataset.\nFound in gr00t/data/embodiment_tags.py. Default is "baxter".',
        default='new_embodiment',
        choices=[embd.value for embd in EmbodimentTag._member_map_.values()]
    )
    parser.add_argument('-dc', '--data_config',
        help='The embodiment tag that is used to finetune the dataset.\nFound in gr00t/data/embodiment_tags.py. Default is "baxter_arms".',
        choices=[config for config in DATA_CONFIG_MAP.keys()],
        default='baxter_arms'
    )
    parser.add_argument('-d', '--dataset', 
        help="The dataset that should be used to calculate the MSE",
        default=DATASET_PATH
    )
    parser.add_argument('-ckpt', '--checkpoint',
        help="The model checkpoint that should be used for the policy",
        default="models/gr00t-1/finetuned-model/checkpoint-500/"
    )
    parser.add_argument('-p', '--policy',
        help="Filename for the gesture to be saved - will be saved in Gestures/policy",
    )
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    warnings.simplefilter("ignore", category=FutureWarning)
    tasks = read_jsonl(os.path.join(ROOT_DIR, args.dataset, 'meta/episodes.jsonl'))
    
    # Get the target data_config class for the modalities and transforms
    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    # Instantiate the finetuned version of gr00t's policy, using the model and target embodiment
    finetuned_policy = Gr00tPolicy(
        model_path=args.checkpoint,
        embodiment_tag=args.embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=device,
    )
    for key, value in modality_config.items():
        if isinstance(value, np.ndarray):
            print(key, value.shape)
        else:
            print(key, value)
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset,
        modality_configs=modality_config,
        video_backend="decord",
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=args.embodiment_tag,
    )

    # Dataset joint order
    joint_order = [
        "left_s0", "left_s1", "left_e0", "left_e1",
        "left_w0", "left_w1", "left_w2",
        "right_s0", "right_s1", "right_e0", "right_e1",
        "right_w0", "right_w1", "right_w2"
    ]

    state_trajectories = []
    gt_trajectories = []
    pred_trajectories = []
    print(dataset.get_trajectory_data(0).shape)
    for i, task in enumerate(tasks):
        state, action, pred = predict_actions(
            policy=finetuned_policy,
            dataset=dataset,
            traj_id=i,
            steps=dataset.get_trajectory_data(i).shape[0],
            modality_keys=['left_arm', 'right_arm'],
            )
        state_trajectories.append(state)
        gt_trajectories.append(action)
        pred_trajectories.append(pred)
        
    for task, trajectory in zip(tasks, trajectories):
        baxter_joints = []
        # Get all waypoints for trajectory
        for action in trajectory:
            assert len(joint_order) == len(action), f"Length joint_order: {len(joint_order)}, Length action: {len(action)}"
            baxter_wp = {key:value for key, value in zip(joint_order, action)}
            baxter_joints.append(baxter_wp)
        
        # Save waypoints to csv
        filepath = f'Gestures/{policy}/{" ".join(task["tasks"][0].split()[:3])}.csv'
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(filepath, 'w') as gestures:
            for row in baxter_joints:
                cssv = ",".join(f'"{key}":{val}' for key, val in row.items())
                gestures.write("{" + cssv + "}\n")

        
