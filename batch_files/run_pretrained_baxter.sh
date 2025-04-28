#!/bin/bash

#SBATCH --job-name=Baxter_Pretrained  # Job name
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --gpus-per-task=1           # Number of CPU cores per task
#SBATCH --mem=32G  # Request more memory (adjust as needed)
#SBATCH --constraint=gpul40s        # Type of gpu required for task
#SBATCH --time=03:00:00             # Time limit hrs:min:sec
#SBATCH --output=batch_files/pretrained_baxter_output_%j.log      # Standard output log
# Note: The above comments are supposed to be like this, its how the compiler registers flags for sbatch

# TO RUN THIS SCRIPT DO: sbatch -A axb1653 -p gpu --constraint=gpul40s fine_tune_gr00t.sh

# Load any modules if needed
# module load python
module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate gr00t
# Run your Python script
python -u scripts/get_trajectory_mse.py -em gr1 -dc baxter_arms -d novideo_data -ckpt nvidia/GR00T-N1-2B -p Pretrained-Policy-Baxter
conda deactivate
