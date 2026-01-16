import robomimic
import os
import torch
from robomimic.config import config_factory
from robomimic.scripts.train import train
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_divergence_loss","-CDM",
        action='store_true',
        help="set this flag to use divergence loss during training"
    )
    parser.add_argument(
        "--div_loss_weight","-L",
        type=float,
        default=0.01,
        help="weight for divergence loss if used"
    )
    parser.add_argument(
        "--dataset_path","-D",
        type=str,
        default="/app/robomimic/datasets/lift/ph/low_dim_v15_w_cdm.hdf5",
        help="path to dataset hdf5 file"
    )
    return parser.parse_args()

args = parse_args()

# Path to your dataset with divergence info
# Update this path if your file is located elsewhere
dataset_path = os.path.expanduser(args.dataset_path)

# Create default BC configuration
config = config_factory(algo_name="bc")

with config.values_unlocked():
    # Set dataset path
    config.train.data = dataset_path
    
    # Set output directory for results
    base_dir = "./exps/results/bc_rss/mlp"
    if args.use_divergence_loss:
        base_dir += "_divergence"
        cdm_weight = args.div_loss_weight
    else:
        base_dir += "_no_divergence"
        cdm_weight = 0.0
    config.train.output_dir = os.path.expanduser(base_dir)

    # search the output directory for existing experiments to set experiment name
    try:
        existing_exps = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("exp")]
        exp_nums = [int(d.split("exp")[-1]) for d in existing_exps if d.split("exp")[-1].isdigit()]
        exp_num = max(exp_nums) + 1 if exp_nums else 1
    except FileNotFoundError:
        exp_num = 1

    # config.experiment.name = "no_divergence"
    config.experiment.name = f"exp{exp_num}"

    # Configure observation keys
    # CRITICAL: 'robot0_eef_pos' and 'robot0_eef_quat' are required for 
    # the divergence computation (div_v_t) in the loss function.
    config.observation.modalities.obs.low_dim = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "object",
    ]
    
    # Disable RNN/Transformer to ensure we are training an MLP
    config.algo.rnn.enabled = False
    config.algo.transformer.enabled = False
    
    # MLP architecture settings (standard BC-MLP)
    config.algo.actor_layer_dims = [1024, 1024]
    
    # NEW: Set divergence loss weight
    config.algo.loss.cdm_weight = cdm_weight

    # Training settings
    config.train.batch_size = 256
    config.train.num_epochs = 200
    config.train.cuda = torch.cuda.is_available()
    
    # Save checkpoints
    config.experiment.save.enabled = True
    config.experiment.save.every_n_epochs = 50
    
    # Validation settings (disable to keep it simple for now)
    config.experiment.validate = False 

# Print config to verify
print("Training Configuration:")
print(config)

# Run training
train(config, device="cuda" if torch.cuda.is_available() else "cpu")
