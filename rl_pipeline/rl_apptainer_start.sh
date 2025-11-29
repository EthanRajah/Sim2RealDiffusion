#!/bin/bash

export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export PATH=/usr/local/cuda-12.6/bin/:$PATH
export CUDA_HOME=/usr/local/cuda
export DIFFUSION_MODEL='/src/instaflow_solid_model/2000'
export DR_YAML_PATH='/src/DiffusionPushBlock_no_dr.yaml'
export UNITY_ENV_PATH='/src/PushBlock_Build_Reward/pushblock_solid_dr.x86_64'
export OUTPUT_DIR='/logs/nodr_instaflow'

nvidia-smi
nvcc --version

# Login to Weights & Biases
if [ -n "$WANDB_API_KEY" ]; then
    wandb login $WANDB_API_KEY
else
    echo "Warning: WANDB_API_KEY not set. Wandb logging may fail."
fi
wandb offline

# Echo parameters for logging
echo "=================== Unity RL Pipeline Parameters ==================="
echo "Environment Path: /src/PushBlock_Build_Reward/pushblock_solid_dr.x86_64"
echo "Domain Randomization YAML: $DR_YAML_PATH"
echo "Diffusion Model: $DIFFUSION_MODEL"
echo "Diffusion Prompt: pushblock"
echo "Output Type: img"
echo "ControlNet Conditioning Scales: [1.0, 1.2]"
echo "Guidance Scale: 1.0"
echo "Denoising Steps: 2"
echo "RL Resolution: 64"
echo "Log Directory: $OUTPUT_DIR"
echo "Using Instaflow: True"
echo "===================================================================="

# Run Unity RL pipeline with Instaflow
CUDA_VISIBLE_DEVICES=0 python /src/unity_gym.py \
  --env_path=$UNITY_ENV_PATH \
  --dr_yaml_path=$DR_YAML_PATH \
  --diffusion_model=$DIFFUSION_MODEL \
  --diffusion_prompt='pushblock' \
  --log_dir=$OUTPUT_DIR \
  --out_type='img' \
  --control_condition 1.0 1.2 \
  --guidance_scale=1.0 \
  --denoise=2 \
  --instaflow \
  --resume


