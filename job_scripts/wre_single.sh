#!/bin/bash
#SBATCH --job-name=wre
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=physical-gpu
#SBATCH --mem=32G
#SBATCH --output=logs/%j-wre.out

ENV_FILE=$1

if [ -z "$ENV_FILE" ]; then
    echo "Error: No environment file specified"
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Environment file $ENV_FILE not found"
    exit 1
fi

# Load the specified environment file
source "$ENV_FILE"

# Build the command
CMD="python ../wr_experiment.py \
    --model \"$MODEL\" \
    --target-lang \"$TARGET_LANG\" \
    --save-folder \"$SAVE_FOLDER\" \
    --data \"$DATA\" \
    --prompt \"$PROMPT\" \
    --hidden-base \"$HIDDEN_BASE\" \
    --hidden-target \"$HIDDEN_TARGET\""

# Add data-sample-size only if set
if [ -n "$DATA_SAMPLE_SIZE" ]; then
    CMD+=" --data-sample-size \"$DATA_SAMPLE_SIZE\""
fi

# Create the artifact folder
mkdir -p "$SAVE_FOLDER"

export PYTHONUNBUFFERED=1

# Run the command
eval $CMD

echo "WRE SINGLE JOB DONE"