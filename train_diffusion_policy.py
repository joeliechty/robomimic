import robomimic
import os
import torch
from robomimic.config import config_factory
from robomimic.scripts.train import train
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", "-D",
        type=str,
        default="lift",
        help="path to dataset hdf5 file"
    )
    parser.add_argument(
        "--output_dir", "-O",
        type=str,
        default="./exps/results/bc_rss/diffusion_policy",
        help="directory to save results"
    )
    parser.add_argument(
        "--num_epochs", "-E",
        type=int,
        default=500,
        help="number of training epochs"
    )
    parser.add_argument(
        "--batch_size", "-B",
        type=int,
        default=256,
        help="training batch size"
    )
    return parser.parse_args()

args = parse_args()

if args.dataset == "lift":
    args.dataset_path = "/app/robomimic/datasets/lift/ph/low_dim_v15.hdf5"
elif args.dataset == "can":
    args.dataset_path = "/app/robomimic/datasets/can/img/can_feat.hdf5"
elif args.dataset == "square":
    args.dataset_path = "/app/robomimic/datasets/square/img/square_feat.hdf5"
else:
    raise ValueError(f"Unknown dataset {args.dataset}. Please specify one of 'lift', 'can', or 'square'.")

# Path to your dataset
dataset_path = os.path.expanduser(args.dataset_path)

# Create Diffusion Policy configuration
config = config_factory(algo_name="diffusion_policy")

with config.values_unlocked():
    # Set dataset path
    config.train.data = dataset_path
    
    # Set output directory for results
    base_dir = os.path.expanduser(args.output_dir)
    base_dir = os.path.join(base_dir, args.dataset)
    config.train.output_dir = base_dir

    # Search the output directory for existing experiments to set experiment name
    try:
        existing_exps = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("exp")]
        exp_nums = [int(d.split("exp")[-1]) for d in existing_exps if d.split("exp")[-1].isdigit()]
        exp_num = max(exp_nums) + 1 if exp_nums else 1
    except FileNotFoundError:
        exp_num = 1

    config.experiment.name = f"exp{exp_num}"

    # Configure observation keys
    config.observation.modalities.obs.low_dim = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "object",
    ]
    
    # Horizon parameters (key parameters for Diffusion Policy)
    config.algo.horizon.observation_horizon = 2  # number of observation frames to condition on
    config.algo.horizon.action_horizon = 8       # number of actions to execute during rollout
    config.algo.horizon.prediction_horizon = 16  # number of actions to predict
    
    # Make sure seq_length and frame_stack match
    config.train.seq_length = config.algo.horizon.prediction_horizon
    config.train.frame_stack = config.algo.horizon.observation_horizon
    
    # UNet architecture (default settings are good)
    config.algo.unet.enabled = True
    config.algo.unet.diffusion_step_embed_dim = 256
    config.algo.unet.down_dims = [256, 512, 1024]
    config.algo.unet.kernel_size = 5
    config.algo.unet.n_groups = 8
    
    # EMA (Exponential Moving Average) for stable training
    config.algo.ema.enabled = True
    config.algo.ema.power = 0.75
    
    # Noise scheduler - DDPM by default
    config.algo.ddpm.enabled = True
    config.algo.ddpm.num_train_timesteps = 100
    config.algo.ddpm.num_inference_timesteps = 100
    config.algo.ddpm.beta_schedule = 'squaredcos_cap_v2'
    config.algo.ddpm.clip_sample = True
    config.algo.ddpm.prediction_type = 'epsilon'
    
    # Optimizer settings
    config.algo.optim_params.policy.optimizer_type = "adamw"
    config.algo.optim_params.policy.learning_rate.initial = 1e-4
    config.algo.optim_params.policy.learning_rate.scheduler_type = "cosine"
    config.algo.optim_params.policy.learning_rate.num_cycles = 0.5
    config.algo.optim_params.policy.learning_rate.warmup_steps = 500
    config.algo.optim_params.policy.regularization.L2 = 1e-6

    # Training settings
    config.train.batch_size = args.batch_size
    config.train.num_epochs = args.num_epochs
    config.train.cuda = torch.cuda.is_available()
    
    # Save checkpoints
    config.experiment.save.enabled = True
    config.experiment.save.every_n_epochs = 10
    
    # Validation settings (disable to keep it simple for now)
    config.experiment.validate = False 

# Print config to verify
print("Training Configuration:")
print(config)

# Run training
train(config, device="cuda" if torch.cuda.is_available() else "cpu")
