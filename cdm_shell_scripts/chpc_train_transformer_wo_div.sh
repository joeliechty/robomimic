#!/bin/bash

# Check if arguments were passed
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ] || [ -z "$6" ]; then
  echo "Error: You must provide a dataset name, dataset portion, portion ID, number of epochs, and save frequency."
  echo "Usage: bash train_transformer_w_div.sh <DATASET> <PORTION> <PORTION_ID> <EPOCHS> <SAVE_FREQ>"
  echo "Example: bash train_transformer_w_div.sh lift full 0 500 20"
  echo "Datasets: lift, can, square"
  echo "Portions: full, half, quarter"
  echo "Portion IDs: 0, 1, 2, ..."
  echo "Epochs: number of training epochs"
  echo "Save Frequency: how often to save the model (in epochs)"
  echo "Batch Size: training batch size"
  exit 1
fi

DATASET=$1
PORTION=$2
PORTION_ID=$3
EPOCHS=$4
SAVE_FREQ=$5
BATCH_SIZE=$6
RESUME=${7:-False}
SEED=${8:-}

if [ "$RESUME" = "True" ]; then
  RESUME_FLAG="--resume"
else
  RESUME_FLAG=""
fi

SEED_ARG=""
if [ -n "$SEED" ]; then
  SEED_ARG="--seed ${SEED}"
fi

# set action chunk size to 16 for tool, transport, and square, else set to 1 for lift and can
if [ "$DATASET" = "tool" ] || [ "$DATASET" = "transport" ] || [ "$DATASET" = "square" ]; then
  ACTION_CHUNK_SIZE=16
else
  ACTION_CHUNK_SIZE=1
fi

# Define your Apptainer image path on scratch
IMAGE_PATH="/scratch/general/vast/$USER/robomimic.sif"

# Load Apptainer module
module load apptainer

echo "Launching Transformer NO CDM job for ${DATASET} with portion: ${PORTION}, epochs: ${EPOCHS}, save frequency: ${SAVE_FREQ}, action chunk size: ${ACTION_CHUNK_SIZE}, seed: ${SEED}"
echo "Start time: $(date)"

# Launch Transformer WITHOUT Divergence
apptainer exec \
  --nv \
  --bind $(pwd):/app/robomimic \
  --pwd /app/robomimic \
  $IMAGE_PATH \
  /bin/bash -c "export NUMBA_CACHE_DIR=/tmp && export PYTHONUSERBASE=/scratch/general/vast/$USER/.local && source /opt/conda/etc/profile.d/conda.sh && conda activate robomimic_venv && pip install --user -e . && python train_divergence_transformer_images.py -D ${DATASET} -DP ${PORTION} -PI ${PORTION_ID} -E ${EPOCHS} -SF ${SAVE_FREQ} -E2E -B ${BATCH_SIZE} -V ${RESUME_FLAG} ${SEED_ARG} -ACS ${ACTION_CHUNK_SIZE}"

echo "Job finished for ${DATASET} with portion: ${PORTION}, epochs: ${EPOCHS}, save frequency: ${SAVE_FREQ}"