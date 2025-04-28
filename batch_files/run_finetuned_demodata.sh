#!/bin/bash

#SBATCH --job-name=Demo_FineTuned_Policy  # Job name
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --gpus-per-task=1           # Number of CPU cores per task
#SBATCH --mem=32G  # Request more memory (adjust as needed)
#SBATCH --constraint=gpul40s        # Type of gpu required for task
#SBATCH --time=03:00:00             # Time limit hrs:min:sec
#SBATCH --output=batch_files/finetuned_demodata_output_%j.log      # Standard output log
# Note: The above comments are supposed to be like this, its how the compiler registers flags for sbatch

# TO RUN THIS SCRIPT DO: sbatch -A axb1653 -p gpu --constraint=gpul40s fine_tune_gr00t.sh

# Load any modules if needed
# module load python
module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate gr00t
# Run your Python script
python scripts/gr00t_finetune.py --dataset-path demo_data/robot_sim.PickNPlace --num-gpus 1 --max-steps 500 --output-dir models/finetuned-demodata-model --data-config gr1_arms_only
python -u scripts/get_trajectory_mse.py -em new_embodiment --data_config gr1_arms_only -d demo_data/robot_sim.PickNPlace -ckpt models/finetuned-demodata-model -p Finetuned-Policy-DemoData
conda deactivate
