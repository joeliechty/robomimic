#!/bin/bash

# Check if arguments were passed
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ] || [ -z "$6" ]; then
  echo "Error: You must provide a dataset name, dataset portion, portion ID, number of epochs, and save frequency."
  echo "Usage: bash train_transformer_w_div.sh <DATASET> <PORTION> <PORTION_ID> <EPOCHS> <SAVE_FREQ>"
  echo "Example: bash train_transformer_w_div.sh lift full 0 500 20 64"
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
CDM_LOSS_WEIGHT=$7
PATIENCE=$8
DECAY_FACTOR=$9
COSINE_REG_SCHEDULE=${10:-False}
RESUME=${11:-False}
COSINE_DECAY_END=${12:-0}
MIN_CDM_WEIGHT=${13:-0.0000001}
SEED=${14:-}


if [ -z "$CDM_LOSS_WEIGHT" ]; then
  CDM_LOSS_WEIGHT=0.001
fi

if [ "$COSINE_REG_SCHEDULE" = "True" ]; then
  COSINE_REG_ARG="-CRS"
else
  COSINE_REG_ARG=""
fi

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

echo "Launching Transformer CDM job for ${DATASET} with portion: ${PORTION}, epochs: ${EPOCHS}, save frequency: ${SAVE_FREQ}, CDM loss weight: ${CDM_LOSS_WEIGHT}, batch size: ${BATCH_SIZE}, patience: ${PATIENCE}, decay factor: ${DECAY_FACTOR}, cosine reg schedule: ${COSINE_REG_SCHEDULE}, resume: ${RESUME}, cosine decay end: ${COSINE_DECAY_END}, min CDM weight: ${MIN_CDM_WEIGHT}, seed: ${SEED}, action chunk size: ${ACTION_CHUNK_SIZE}"
echo "Start time: $(date)"

# Launch Transformer WITH Divergence (-CDM flag)
apptainer exec \
  --nv \
  --bind $(pwd):/app/robomimic \
  --pwd /app/robomimic \
  $IMAGE_PATH \
  /bin/bash -c "export NUMBA_CACHE_DIR=/tmp && export PYTHONUSERBASE=/scratch/general/vast/$USER/.local && source /opt/conda/etc/profile.d/conda.sh && conda activate robomimic_venv && pip install --user -e . && python train_divergence_transformer_images.py -D ${DATASET} -CDM -L ${CDM_LOSS_WEIGHT} -DP ${PORTION} -PI ${PORTION_ID} -E ${EPOCHS} -SF ${SAVE_FREQ} -E2E -B ${BATCH_SIZE} -V --cdm_patience ${PATIENCE} --cdm_decay_factor ${DECAY_FACTOR} ${COSINE_REG_ARG} ${SEED_ARG} ${RESUME_FLAG} --cosine_decay_end ${COSINE_DECAY_END} --min_cdm_weight ${MIN_CDM_WEIGHT} -ACS ${ACTION_CHUNK_SIZE}"

echo "Job finished for ${DATASET} with portion: ${PORTION}, epochs: ${EPOCHS}, save frequency: ${SAVE_FREQ}"