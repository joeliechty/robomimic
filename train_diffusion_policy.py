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
    parser.add_argument(
        "--dataset_portion","-DP",
        type=str,
        default="full",
        choices=["full", "half", "quarter"],
        help="dataset portion: 'full', 'half', or 'quarter'"
    )
    parser.add_argument(
        "--portion_id","-PI",
        type=int,
        default=1,
        help="which portion (1-2 for half, 1-4 for quarter, ignored for full)"
    )
    parser.add_argument(
        "--save_freq","-SF",
        type=int,
        default=5,
        help="save checkpoint every N epochs"
    )
    return parser.parse_args()

args = parse_args()

if args.dataset_portion == "full":
    portion_prefix = "F"
    dataset_suffix = ""
elif args.dataset_portion == "half":
    portion_prefix = f"H{args.portion_id}"
    dataset_suffix = f"_H{args.portion_id}"
elif args.dataset_portion == "quarter":
    portion_prefix = f"Q{args.portion_id}"
    dataset_suffix = f"_Q{args.portion_id}"
else:
    portion_prefix = "F"
    dataset_suffix = ""

if args.dataset == "lift":
    args.dataset_path = f"/app/robomimic/datasets/lift/low_dim_v15{dataset_suffix}.hdf5"
elif args.dataset == "can":
    args.dataset_path = f"/app/robomimic/datasets/can/can_feats{dataset_suffix}.hdf5"
elif args.dataset == "square":
    args.dataset_path = f"/app/robomimic/datasets/square/square_feats{dataset_suffix}.hdf5"
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

    # Convert to absolute path to avoid issues with relative paths
    abs_base_dir = os.path.abspath(os.path.join("robomimic", base_dir))
    
    # search the output directory for existing experiments to set experiment name
    exp_num = 0
    print(f"Checking existing experiments in {abs_base_dir} to set experiment name...")
    if os.path.exists(abs_base_dir):
        print("Found existing base directory.")
        all_items = os.listdir(abs_base_dir)
        print(f"All items in directory: {all_items}")
        existing_experiments = [d for d in all_items if os.path.isdir(os.path.join(abs_base_dir, d)) and d.startswith("exp")]
        print(f"Filtered experiment directories: {existing_experiments}")
        if existing_experiments:
            print(f"Found existing experiments: {existing_experiments}")
            # Extract numbers from experiment folder names (e.g., "exp0" -> 0, "exp1" -> 1)
            exp_numbers = []
            for exp_dir in existing_experiments:
                try:
                    num = int(exp_dir.replace("exp", ""))
                    exp_numbers.append(num)
                except ValueError:
                    continue
            if exp_numbers:
                exp_num = max(exp_numbers) + 1

    config.train.output_dir = base_dir

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
    config.experiment.save.every_n_epochs = args.save_freq
    
    # Set experiment name with dataset portion, epochs, and save frequency
    config.experiment.name = f"{portion_prefix}_{args.num_epochs}_{args.save_freq}_exp{exp_num}"
    
    # Validation settings (disable to keep it simple for now)
    config.experiment.validate = False 

    # Rollout settings
    config.experiment.rollout.enabled = False 

# Print config to verify
print("Training Configuration:")
print(config)

# Run training
train(config, device="cuda" if torch.cuda.is_available() else "cpu")
