#!/bin/bash

# Check if arguments were passed
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ]; then
  echo "Error: You must provide a task name, dataset portion, training epochs, save frequency, and seed, optional are start epoch, end epoch, eval frequency, and video flag."
  echo "Usage: bash loop_eval_transformer_wo_div.sh <TASK> <DATASET_PORTION> <TRAINING_EPOCHS> <SAVE_FREQ> <SEED> [START_EPOCH] [END_EPOCH] [EVAL_FREQ] [VIDEO_FLAG]"
  echo "Example: bash loop_eval_transformer_wo_div.sh lift full 500 20 0 100 500 40 V"
  echo "Tasks: lift, can, square"
  echo "Dataset Portions: F, H1, H2, Q1, Q2, Q3, Q4"
  echo "Training Epochs: number of training epochs"
  echo "Save Frequency: how often the model was saved (in epochs)"
  echo "Seed: random seed for evaluation"
  echo "Start Epoch (optional): first epoch to evaluate (default: save_freq)"
  echo "End Epoch (optional): last epoch to evaluate (default: training_epochs)"
  echo "Eval Frequency (optional): evaluation frequency, must be multiple of save_freq (default: save_freq)"
  echo "Video Flag (optional): V to save video, no flag to skip video"
  exit 1
fi

TASK=$1
DATASET_PORTION=$2
TRAINING_EPOCHS=$3
SAVE_FREQ=$4
SEED=$5
START_EPOCH=$6
END_EPOCH=$7
EVAL_FREQ=$8
VIDEO_FLAG=$9

# Build optional arguments
LOOP_ARGS="-LOOP"

if [ -n "$START_EPOCH" ]; then
  LOOP_ARGS="$LOOP_ARGS -START ${START_EPOCH}"
fi

if [ -n "$END_EPOCH" ]; then
  LOOP_ARGS="$LOOP_ARGS -END ${END_EPOCH}"
fi

if [ -n "$EVAL_FREQ" ]; then
  LOOP_ARGS="$LOOP_ARGS -EF ${EVAL_FREQ}"
fi

if [ -n "$VIDEO_FLAG" ]; then
  LOOP_ARGS="$LOOP_ARGS -V"
fi

# Add images flag for can and square tasks (image-trained models)
IMAGES_FLAG=""
if [ "$TASK" = "can" ] || [ "$TASK" = "square" ]; then
  IMAGES_FLAG="-I"
fi

# Launch Transformer Baseline Evaluation Loop
docker run -d \
  --name loop_eval_transformer_base_${TASK}_${DATASET_PORTION}_${TRAINING_EPOCHS}_${SAVE_FREQ}_seed${SEED} \
  --gpus all \
  --net=host \
  -v $(pwd):/app/robomimic \
  -w /app/robomimic \
  robomimic \
  /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate robomimic_venv && pip install -e . && pip install git+https://github.com/openai/CLIP.git && python eval_rollouts.py -M transformer -T ${TASK} -DS ${DATASET_PORTION} -TE ${TRAINING_EPOCHS} -SF ${SAVE_FREQ} -S ${SEED} -SD ${IMAGES_FLAG} ${LOOP_ARGS}"

echo "Launched Transformer baseline evaluation loop job for ${TASK} with dataset portion: ${DATASET_PORTION}, training epochs: ${TRAINING_EPOCHS}, save frequency: ${SAVE_FREQ}, seed: ${SEED}"
