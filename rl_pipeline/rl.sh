#!/bin/bash
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=2    # There are 6 CPU cores per 3g.20gb and 4g.20gb on Narval.
#SBATCH --mem=24gb           # There are 62GB GPU RAM per 3g.20gb and 4g.20gb on Narval.
#SBATCH --time=7-00:00:00
 
module --force purge
module load gentoo/2020 # Prevent GLIBC_ABI_DT_RELR error
module load apptainer

# Set Weights & Biases API key
export WANDB_API_KEY=$(cat /home/rajaheth/projects/def-lakahrs/rajaheth/rl_training/.wandb_api_key)
 
xvfb-run vglrun -d egl apptainer exec --nv \
  --env WANDB_API_KEY=$WANDB_API_KEY \
  --bind ~/projects/def-lakahrs/rajaheth/rl_training/src:/src \
  --bind ~/projects/def-lakahrs/rajaheth/rl_training/logs:/logs \
  rl_pipeline.sif bash rl_apptainer_start.sh
