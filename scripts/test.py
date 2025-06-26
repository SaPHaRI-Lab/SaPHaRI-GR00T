from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.data.dataset import LeRobotSingleDataset
import os

if __name__ == "__main__":
    DATASET_PATH = os.path.relpath('novideo_data/')
    print(DATASET_PATH)
    data_config = DATA_CONFIG_MAP['baxter_arms']
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    dataset = LeRobotSingleDataset(
        dataset_path=DATASET_PATH,
        modality_configs=modality_config,
        video_backend="decord",
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag='baxter',
    )
    print("All steps:", len(dataset.all_steps))   # pick the failing sample
    for traj_id in range(4):
        df        = dataset.get_trajectory_data(traj_id)
        true_len  = len(df)                # rows in parquet
        meta_len  = dataset.trajectory_lengths[
                        dataset.get_trajectory_index(traj_id)]

        print(traj_id, true_len, meta_len)