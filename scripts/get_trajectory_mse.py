from gr00t.utils.eval import calc_mse_for_single_trajectory
import warnings
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.schema import EmbodimentTag
from gr00t.data.dataset import LeRobotSingleDataset
import numpy as np
import torch
import argparse
PRE_TRAINED_MODEL_PATH = "nvidia/GR00T-N1-2B"
DATASET_PATH = "../demo_data/robot_sim.PickNPlace"

if __name__ = "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-em', '--embodiment_tag', 
        help='The embodiment tag that is used to finetune the dataset.\nFound in gr00t/data/embodiment_tags.py. Default is "baxter".',
        choices=[tags for tags in EmbodimentTag._member_names_],
        default='baxter',
    )
    parser.add_argument('-dc', '--data_config', 
        help='The embodiment tag that is used to finetune the dataset.\nFound in gr00t/data/embodiment_tags.py. Default is "baxter_arms".',
        choices=[tags for tags in EmbodimentTag._member_names_],
        default='baxter_arms',
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    warnings.simplefilter("ignore", category=FutureWarning)
    
    # Get the target data_config class for the modalities and transforms
    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    
    # Instantiate the finetuned version of gr00t's policy, using the model and target embodiment 
    finetuned_model_path = "/tmp/gr00t-1/finetuned-model/checkpoint-500"
    finetuned_policy = Gr00tPolicy(
        model_path=finetuned_model_path,
        embodiment_tag=args.embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=device,
    )

    dataset = LeRobotSingleDataset(
        dataset_path=DATASET_PATH,
        modality_configs=modality_config,
        video_backend="decord",
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=args.embodiment_tag,
    )
    
    mse = calc_mse_for_single_trajectory(
        finetuned_policy,
        dataset,
        traj_id=0,
        modality_keys=["right_arm", "right_hand"],   # we will only evaluate the right arm and right hand
        steps=150,
        action_horizon=16,
        plot=True
    )
    warnings.simplefilter("ignore", category=FutureWarning)

    mse = calc_mse_for_single_trajectory(
        finetuned_policy,
        dataset,
        traj_id=0,
        modality_keys=["right_arm", "right_hand"],   # we will only evaluate the right arm and right hand
        steps=150,
        action_horizon=16,
        plot=True
    )

    print("MSE loss for trajectory 0:", mse)