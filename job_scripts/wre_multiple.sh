#!/bin/bash
#SBATCH --job-name=wre-multiple
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=physical-gpu
#SBATCH --mem=32G
#SBATCH --output=logs/%j-wre-multiple.out

# This script is for MODEL folder path with multiple checkpoint folders

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

export PYTHONUNBUFFERED=1

# Loop over each checkpoint folder inside MODEL
for CHECKPOINT_DIR in "$MODEL"/checkpoint-*; do
    if [ -d "$CHECKPOINT_DIR" ]; then
        # Extract checkpoint folder name (e.g., checkpoint-3760)
        CHECKPOINT_NAME=$(basename "$CHECKPOINT_DIR")

        # Create a subfolder for this checkpoint inside SAVE_FOLDER
        CHECKPOINT_SAVE_FOLDER="$SAVE_FOLDER/$CHECKPOINT_NAME"
        mkdir -p "$CHECKPOINT_SAVE_FOLDER"

        echo "Running WRE on $CHECKPOINT_DIR -> saving to $CHECKPOINT_SAVE_FOLDER"
        
        CMD="python ../wr_experiment.py \
        --model \"$CHECKPOINT_DIR\" \
        --target-lang \"$TARGET_LANG\" \
        --save-folder \"$CHECKPOINT_SAVE_FOLDER\" \
        --data \"$DATA\" \
        --prompt \"$PROMPT\" \
        --hidden-base \"$HIDDEN_BASE\" \
        --hidden-target \"$HIDDEN_TARGET\""
        
        # Add data-sample-size only if set
        if [ -n "$DATA_SAMPLE_SIZE" ]; then
            CMD+=" --data-sample-size \"$DATA_SAMPLE_SIZE\""
        fi
        
        # Run the command
        eval $CMD
        
        echo "Finished WRE on $CHECKPOINT_DIR"
    fi
done

echo "WRE ALL CHECKPOINTS DONE"
