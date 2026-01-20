#!/bin/bash

# Check if arguments were passed
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: You must provide a job ID number (used as seed) and dataset name."
  echo "Usage: bash eval_vae_w_div_docker.sh <ID> <DATASET>"
  echo "Example: bash eval_vae_w_div_docker.sh 1 lift"
  echo "Datasets: lift, can, square"
  exit 1
fi

JOB_ID=$1
DATASET=$2
SEED=$JOB_ID

# Launch VAE CDM Evaluation
docker run -d \
  --name eval_vae_cdm_${DATASET}_${JOB_ID} \
  --gpus all \
  --net=host \
  -v $(pwd):/app/robomimic \
  -w /app/robomimic \
  robomimic \
  /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate robomimic_venv && pip install -e . && python eval_model.py --model vae --divergence --task ${DATASET} --exp 0 --epoch 1000 --n_rollouts 50 --seed ${SEED} --video --save_data"

echo "Launched VAE CDM evaluation job for ${DATASET} with Seed: ${SEED}"
