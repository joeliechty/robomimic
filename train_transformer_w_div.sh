#!/bin/bash

# Check if arguments were passed
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: You must provide a job ID number and dataset name."
  echo "Usage: bash train_transformer_w_div.sh <ID> <DATASET>"
  echo "Example: bash train_transformer_w_div.sh 1 lift"
  echo "Datasets: lift, can, square"
  exit 1
fi

JOB_ID=$1
DATASET=$2

# Launch Transformer WITH Divergence (-CDM flag)
docker run -d \
  --name train_transformer_cdm_${DATASET}_${JOB_ID} \
  --gpus all \
  --net=host \
  -v $(pwd):/app/robomimic \
  -w /app/robomimic \
  robomimic \
  /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate robomimic_venv && pip install -e . && python train_divergence_transformer.py -D ${DATASET} -CDM -L 0.01 -E 1000"

echo "Waiting 20s to prevent experiment ID collision..."
sleep 20

echo "Launched transformer CDM job for ${DATASET} with ID: ${JOB_ID}"