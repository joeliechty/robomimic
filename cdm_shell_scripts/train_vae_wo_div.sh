#!/bin/bash

# Check if arguments were passed
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: You must provide a dataset name, dataset portion, portion ID, number of epochs, and save frequency."
  echo "Usage: bash train_vae_wo_div.sh <DATASET> <PORTION> <PORTION_ID> <EPOCHS> <SAVE_FREQ>"
  echo "Example: bash train_vae_wo_div.sh lift full 0 500 20"
  echo "Datasets: lift, can, square"
  echo "Portions: full, half, quarter"
  echo "Portion IDs: 0, 1, 2, ..."
  echo "Epochs: number of training epochs"
  echo "Save Frequency: how often to save the model (in epochs)"
  exit 1
fi

DATASET=$1
PORTION=$2
PORTION_ID=$3
EPOCHS=$4
SAVE_FREQ=$5

# Launch VAE Baseline (No Divergence)
if [ "$DATASET" = "can" ] || [ "$DATASET" = "square" ]; then
  # Use image-based training for can and square datasets
  docker run -d \
    --name train_vae_base_${DATASET}_${PORTION}_${PORTION_ID}_${EPOCHS}_${SAVE_FREQ} \
    --gpus all \
    --net=host \
    -v $(pwd):/app/robomimic \
    -w /app/robomimic \
    robomimic \
    /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate robomimic_venv && pip install -e . && python train_divergence_vae_image.py -D ${DATASET} -DP ${PORTION} -PI ${PORTION_ID} -E ${EPOCHS} -SF ${SAVE_FREQ} -I"
else
  # Use standard training for other datasets (e.g., lift)
  docker run -d \
    --name train_vae_base_${DATASET}_${PORTION}_${PORTION_ID}_${EPOCHS}_${SAVE_FREQ} \
    --gpus all \
    --net=host \
    -v $(pwd):/app/robomimic \
    -w /app/robomimic \
    robomimic \
    /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate robomimic_venv && pip install -e . && python train_divergence_vae.py -D ${DATASET} -DP ${PORTION} -PI ${PORTION_ID} -E ${EPOCHS} -SF ${SAVE_FREQ}"
fi

echo "Waiting 20s to prevent experiment ID collision..."
sleep 20

echo "Launched VAE baseline job for ${DATASET} with ID: ${JOB_ID}"
