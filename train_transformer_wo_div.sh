
#!/bin/bash

# Check if arguments were passed
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: You must provide a job ID number and dataset name."
  echo "Usage: bash train_transformer_wo_div.sh <ID> <DATASET>"
  echo "Example: bash train_transformer_wo_div.sh 1 lift"
  echo "Datasets: lift, can, square"
  exit 1
fi

JOB_ID=$1
DATASET=$2

# Launch Transformer Baseline (No Divergence)
docker run -d \
  --name train_transformer_base_${DATASET}_${JOB_ID} \
  --gpus all \
  --net=host \
  -v $(pwd):/app/robomimic \
  -w /app/robomimic \
  robomimic \
  /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate robomimic_venv && pip install -e . && python train_divergence_transformer.py -D ${DATASET} -E 2000"

echo "Waiting 20s to prevent experiment ID collision..."
sleep 20

echo "Launched transformer baseline job for ${DATASET} with ID: ${JOB_ID}"
