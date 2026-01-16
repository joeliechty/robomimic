#!/bin/bash

# Check if an ID number was passed
if [ -z "$1" ]; then
  echo "Error: You must provide a job ID number."
  echo "Usage: bash train_transformer_w_div.sh <ID>"
  exit 1
fi

# Launch Transformer WITH Divergence (-CDM flag)
docker run -d \
  --name train_transformer_cdm_$1 \
  --gpus all \
  --net=host \
  -v $(pwd):/app/robomimic \
  -w /app/robomimic \
  robomimic \
  /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate robomimic_venv && pip install -e . && python train_divergence_transformer.py -CDM -L 0.1 -E 500"

echo "Waiting 10s to prevent experiment ID collision..."
sleep 10

echo "Launched transformer CDM job with ID: $1"