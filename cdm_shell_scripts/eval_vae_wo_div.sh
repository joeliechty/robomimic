#!/bin/bash

# Check if arguments were passed
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ] || [ -z "$6" ]; then
  echo "Error: You must provide a task name, dataset portion, training epochs, save frequency, evaluation epoch, and seed, optional is video flag."
  echo "Usage: bash eval_vae_wo_div.sh <TASK> <DATASET_PORTION> <TRAINING_EPOCHS> <SAVE_FREQ> <EVAL_EPOCH> <SEED> <VIDEO_FLAG>"
  echo "Example: bash eval_vae_wo_div.sh lift full 500 20 160 0 V"
  echo "Tasks: lift, can, square"
  echo "Dataset Portions: F, H1, H2, Q1, Q2, Q3, Q4"
  echo "Training Epochs: number of training epochs"
  echo "Save Frequency: how often the model was saved (in epochs)"
  echo "Evaluation Epoch: which epoch to evaluate"
  echo "Seed: random seed for evaluation"
  echo "Video Flag: V to save video, no flag to skip video"
  exit 1
fi

TASK=$1
DATASET_PORTION=$2
TRAINING_EPOCHS=$3
SAVE_FREQ=$4
EVAL_EPOCH=$5
SEED=$6
VIDEO_FLAG=$7

# Conditionally add video flag
if [ -n "$VIDEO_FLAG" ]; then
  VIDEO_ARG="-V"
else
  VIDEO_ARG=""
fi

# Launch VAE Baseline Evaluation
docker run -d \
  --name eval_vae_base_${TASK}_${DATASET_PORTION}_${TRAINING_EPOCHS}_${SAVE_FREQ}_eval${EVAL_EPOCH}_seed${SEED} \
  --gpus all \
  --net=host \
  -v $(pwd):/app/robomimic \
  -w /app/robomimic \
  robomimic \
  /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate robomimic_venv && pip install -e . && python eval_model.py -M vae -T ${TASK} -DS ${DATASET_PORTION} -TE ${TRAINING_EPOCHS} -SF ${SAVE_FREQ} -EE ${EVAL_EPOCH} -S ${SEED} -SD ${VIDEO_ARG}"

echo "Launched VAE baseline evaluation job for ${TASK} with dataset portion: ${DATASET_PORTION}, training epochs: ${TRAINING_EPOCHS}, save frequency: ${SAVE_FREQ}, evaluation epoch: ${EVAL_EPOCH}, seed: ${SEED}"
