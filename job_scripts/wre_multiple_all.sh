#!/bin/bash
#SBATCH --job-name=wre-multiple-all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=physical-gpu
#SBATCH --mem=32G
#SBATCH --output=logs/%j-wre-multiple-all.out

# Base folder containing environment files
ENV_FOLDER="envs/multiple"

# Path to the script that runs WRE for one environment (your previous script)
WRE_SCRIPT="wre_multiple.sh"

# Find all .sh files recursively and submit a WRE job for each
find "$ENV_FOLDER" -type f -name "*.sh" | while read -r ENV_FILE; do
    echo "Submitting WRE job for environment: $ENV_FILE"
    bash "$WRE_SCRIPT" "$ENV_FILE"
done

echo "All WRE jobs submitted for all environment scripts."
