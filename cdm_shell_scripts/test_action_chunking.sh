#!/bin/bash

# Test script for action chunking with transformer BC + CDM
# This is a quick test script to verify the action chunking implementation works

# Check if arguments were passed
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Error: You must provide dataset name, action chunk size, and use_cdm flag."
  echo "Usage: bash test_action_chunking.sh <DATASET> <CHUNK_SIZE> <USE_CDM> [BATCH_SIZE] [EPOCHS] [SEED]"
  echo "Example: bash test_action_chunking.sh lift 4 True 16 10 42"
  echo "Datasets: lift, can, square"
  echo "Chunk Size: 1 (no chunking), 2, 4, 8, etc."
  echo "Use CDM: True or False"
  exit 1
fi

DATASET=$1
CHUNK_SIZE=$2
USE_CDM=$3
BATCH_SIZE=${4:-16}
EPOCHS=${5:-10}
SEED=${6:-42}
SAVE_FREQ=5  # Save every 5 epochs for testing

# CDM parameters (only used if USE_CDM=True)
CDM_LOSS_WEIGHT=0.0001
TIME_WEIGHTING="uniform"  # Options: uniform, exponential, linear
TIME_DECAY=0.5  # Only used for exponential weighting

# Build command arguments
CMD_ARGS="-D ${DATASET} -E ${EPOCHS} -SF ${SAVE_FREQ} -B ${BATCH_SIZE} -ACS ${CHUNK_SIZE} --seed ${SEED} -V -E2E"

# Add CDM flag if requested
if [ "$USE_CDM" = "True" ]; then
  CMD_ARGS="${CMD_ARGS} -CDM -L ${CDM_LOSS_WEIGHT}"
  echo "Testing Action Chunking WITH CDM"
else
  echo "Testing Action Chunking WITHOUT CDM"
fi

# Define your Apptainer image path on scratch
IMAGE_PATH="/scratch/general/vast/$USER/robomimic.sif"

# Load Apptainer module
module load apptainer

echo "===================================================================="
echo "ACTION CHUNKING TEST"
echo "===================================================================="
echo "Dataset: ${DATASET}"
echo "Chunk Size: ${CHUNK_SIZE}"
echo "Use CDM: ${USE_CDM}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Epochs: ${EPOCHS}"
echo "Seed: ${SEED}"
echo "Time Weighting: ${TIME_WEIGHTING}"
if [ "$USE_CDM" = "True" ]; then
  echo "CDM Loss Weight: ${CDM_LOSS_WEIGHT}"
fi
echo "===================================================================="
echo "Start time: $(date)"
echo ""

# Launch training
apptainer exec \
  --nv \
  --bind $(pwd):/app/robomimic \
  --pwd /app/robomimic \
  $IMAGE_PATH \
  /bin/bash -c "export NUMBA_CACHE_DIR=/tmp && export PYTHONUSERBASE=/scratch/general/vast/$USER/.local && source /opt/conda/etc/profile.d/conda.sh && conda activate robomimic_venv && pip install --user -e . && python train_divergence_transformer_images.py ${CMD_ARGS}"

echo ""
echo "===================================================================="
echo "Test finished for ${DATASET} with chunk size ${CHUNK_SIZE}"
echo "End time: $(date)"
echo "===================================================================="
