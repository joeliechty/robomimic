#!/bin/bash

# Evaluate Diffusion Policy models with standardized settings
# 50 rollouts, epoch 1000, videos saved, seed 0

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: bash eval_diffusion.sh <TASK> <EXP_NUM>"
  echo ""
  echo "Arguments:"
  echo "  TASK: lift, can, or square"
  echo "  EXP_NUM: experiment number (e.g., 1 for exp1)"
  echo ""
  echo "Examples:"
  echo "  bash eval_diffusion.sh lift 1"
  echo "  bash eval_diffusion.sh can 0"
  exit 1
fi

TASK=$1
EXP_NUM=$2
SEED=$3

echo "Evaluating Diffusion Policy on ${TASK} task, exp${EXP_NUM}, seed ${SEED}"
echo "Parameters: 50 rollouts, epoch 1000, video and data enabled"
echo ""

python eval_model.py \
  -m diffusion \
  -t ${TASK} \
  -e ${EXP_NUM} \
  -p 1000 \
  -n 50 \
  -s ${SEED} \
  --video \
  --save_data

echo ""
echo "Evaluation complete! Check eval_data/ for stats JSON and eval_videos/ for video."
