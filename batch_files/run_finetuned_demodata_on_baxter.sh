#!/bin/bash

#SBATCH --job-name=finetuned_demodata_on_baxter  # Job name
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --gpus-per-task=1           # Number of CPU cores per task
#SBATCH --mem=32G  # Request more memory (adjust as needed)
#SBATCH --constraint=gpul40s        # Type of gpu required for task
#SBATCH --time=12:00:00             # Time limit hrs:min:sec
#SBATCH --output=batch_files/finetuned_demodata_on_baxter_output.log      # Standard output log
# Note: The above comments are supposed to be like this, its how the compiler registers flags for sbatch

# TO RUN THIS SCRIPT DO: sbatch -A axb1653 -p gpu --constraint=gpul40s batch_files/run_finetuned_demodata_on_baxter.sh

# Load any modules if needed
# module load python
module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate gr00t
# Run your Python script
# python scripts/gr00t_finetune.py --dataset-path demo_data/robot_sim.PickNPlace --num-gpus 1 --max-steps 500 --output-dir models/finetuned-demodata-model --data-config gr1_arms_only
python -u scripts/get_trajectory_mse.py -em new_embodiment --data_config baxter_arms -d novideo_data -ckpt models/finetuned-demodata-model -p Finetuned-DemoData-Policy-On-Baxter
conda deactivate
