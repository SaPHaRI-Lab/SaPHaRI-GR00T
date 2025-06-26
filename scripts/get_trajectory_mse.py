import warnings
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy, BasePolicy
from gr00t.data.schema import EmbodimentTag
from gr00t.data.dataset import LeRobotSingleDataset
import numpy as np
from tqdm import tqdm
import torch
import argparse, matplotlib.pyplot as plt
import json, os
PRE_TRAINED_MODEL_PATH = "nvidia/GR00T-N1-2B"
DATASET_PATH = "novideo_data/"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def calc_mse_for_single_trajectory(
    policy: BasePolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    modality_keys: list,
    steps=600,
    action_horizon=16,
    plot=True,
    save=True,
    task={},
    policy_name="Fine-tuned"
):
    state_joints_across_time = []
    gt_action_joints_across_time = []
    pred_action_joints_across_time = []

    steps = dataset.get_trajectory_data(traj_id).shape[0]
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

    # plot the joints
    state_joints_across_time = np.array(state_joints_across_time)
    gt_action_joints_across_time = np.array(gt_action_joints_across_time)
    pred_action_joints_across_time = np.array(pred_action_joints_across_time)[:steps]
    assert (
        state_joints_across_time.shape
        == gt_action_joints_across_time.shape
        == pred_action_joints_across_time.shape
    )

    # calc MSE across time
    mse = np.mean((gt_action_joints_across_time - pred_action_joints_across_time) ** 2)
    print("Unnormalized Action MSE across single traj:", mse)

    num_of_joints = state_joints_across_time.shape[1]

    if plot:
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 2 * num_of_joints))

        # Add a global title showing the modality keys
        fig.suptitle(
            f"Trajectory {traj_id} - Modalities: {', '.join(modality_keys)}",
            fontsize=16,
            color="blue",
        )
        for i, ax in enumerate(axes.flatten()):
            if i == 7:
                break
            ax.plot(state_joints_across_time[:, i], label="state joints")
            ax.plot(gt_action_joints_across_time[:, i], label="gt action joints")
            ax.plot(pred_action_joints_across_time[:, i], label="pred action joints")

            graph_mse = np.mean((gt_action_joints_across_time[:, i] - pred_action_joints_across_time[:, i]) ** 2)
            ax.set_title(f"{' '.join(task['tasks'][0].split()[:3])} - Joint {i} MSE: {graph_mse}")
            ax.set_xlabel('Steps in Trajectory')
            ax.set_ylabel('Rotation in Radians')
            ax.legend()

        last_ax = axes.flatten()[-1]
        last_ax.axis('off')
        last_ax.text(0.5, 0.5, f"Summary: MSE across all trajectories - {mse}.", 
                    ha='center', va='center', fontsize=12, wrap=True)
        plt.tight_layout()
        plt.show()
        if save:
            filepath = f'figures/{policy_name}/{" ".join(task["tasks"][0].split()[:3])}'
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(filepath)

    return mse

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
        choices=[embd.value for embd in EmbodimentTag._member_map_.values()]
    )
    parser.add_argument('-dc', '--data_config',
        help='The embodiment tag that is used to finetune the dataset.\nFound in gr00t/data/embodiment_tags.py. Default is "baxter_arms".',
        choices=[config for config in DATA_CONFIG_MAP.keys()],
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
        help="The name of the policy being used -- used to different files in the figures folder",
        default="Fine-tuned Baxter Policy"
    )
    parser.add_argument('-s', '--save',
        help="Add flag if you want the generated figures to be saved - Saves to the figure/ directory",
        action='store_true',
        default=True
    )
    # parser.add_argument('-f', '--filename',
    #     help="Filename for the figure to be saved - will be saved in figures",
    # )
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
    
    for i, task in enumerate(tasks):
        mse = calc_mse_for_single_trajectory(
            finetuned_policy,
            dataset,
            traj_id=i,
            modality_keys=["right_arm"],   # we will only evaluate the right arm and right hand
            steps=500,
            action_horizon=16,
            plot=True,
            save=args.save,
            task=task,
            policy_name=args.policy
        )

        print(f"MSE loss for trajectory {i}:", mse)
