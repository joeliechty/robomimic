#!/bin/bash
#SBATCH --account=your_account_name
#SBATCH --partition=your_partition_name
#SBATCH --job-name=robomimic_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=train_%j.log

# Run your modified bash script
bash train_transformer_w_div.sh lift full 0 500 20