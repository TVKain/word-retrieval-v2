#!/bin/bash
#SBATCH --job-name=wre-single-all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=physical-gpu
#SBATCH --mem=32G
#SBATCH --output=logs/%j-wre-single-all.out

# Base folder containing single environment files
ENV_FOLDER="envs/single"

# Path to the WRE script
WRE_SCRIPT="wre_single.sh"  # This is your original single-env script

# Find all .sh files recursively and submit a WRE job for each
find "$ENV_FOLDER" -type f -name "*.sh" | while read -r ENV_FILE; do
    echo "Submitting WRE job for environment: $ENV_FILE"
    bash "$WRE_SCRIPT" "$ENV_FILE"
done

echo "All WRE jobs submitted for single environment scripts."
