import robomimic
import os
import torch
from robomimic.config import config_factory
from robomimic.scripts.train import train
import argparse
import h5py

def sync_all_attributes(source_path, target_path):
    print(f"Syncing attributes from {source_path} to {target_path}...")
    
    with h5py.File(source_path, 'r') as f_src, h5py.File(target_path, 'a') as f_tgt:
        # 1. Sync global /data attributes (env_args, etc.)
        if "data" in f_src and "data" in f_tgt:
            for k, v in f_src["data"].attrs.items():
                f_tgt["data"].attrs[k] = v
            print("  [OK] Global 'data' attributes synced.")

        # 2. Sync per-demo attributes (num_samples, model_file, etc.)
        demos = [k for k in f_src["data"].keys() if k.startswith("demo_")]
        for demo in demos:
            if demo in f_tgt["data"]:
                for k, v in f_src[f"data/{demo}"].attrs.items():
                    f_tgt[f"data/{demo}"].attrs[k] = v
            else:
                print(f"  [Warning] {demo} found in source but not in target. Skipping.")
        
        print(f"  [OK] Attributes for {len(demos)} demos synced.")

# Update these paths to your actual local file locations
# source = "dataset/square/square_demo.hdf5"
# target = "dataset/square/square_feats_w_cdm.hdf5"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", "-D",
        type=str,
        default="lift",
        help="dataset name: 'lift', 'can', or 'square'"
    )
    parser.add_argument(
        "--epochs", "-E",
        type=int,
        default=2000,
        help="number of training epochs"
    )
    parser.add_argument(
        "--output_dir", "-O",
        type=str,
        default="./exps/results/bc_rss/diffusion_policy",
        help="directory to save results"
    )
    parser.add_argument(
        "--batch_size", "-B",
        type=int,
        default=256,
        help="training batch size"
    )
    parser.add_argument(
        "--use_images", "-I",
        action='store_true',
        help="set this flag to include image observations"
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
        default=10,
        help="save checkpoint every N epochs"
    )
    parser.add_argument(
        "--end_to_end_image_training", "-E2E",
        action="store_true",
        help="Train image encoders end-to-end using raw RGB observations"
    )
    parser.add_argument(
        "--validate", "-V",
        action="store_true",
        help="set this flag to run validation during training"
    )
    parser.add_argument(
        "--resume",
        action='store_true',
        help="set this flag to resume training from latest checkpoint"
    )
    return parser.parse_args()

args = parse_args()

if args.end_to_end_image_training:
    args.use_images = False

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

if args.dataset in ["lift", "can", "square"]:
    target = f"datasets/{args.dataset}/{args.dataset}_feats{dataset_suffix}_w_cdm.hdf5"
    source = f"datasets/{args.dataset}/{args.dataset}_demo.hdf5"
else:
    raise ValueError(f"Unknown dataset {args.dataset}. Please specify one of 'lift', 'can', or 'square'.")

# if args.dataset == "lift":
#     target = f"/app/robomimic/datasets/lift/lift_feats{dataset_suffix}_w_cdm.hdf5"
#     source = f"/app/robomimic/datasets/lift/lift_demo.hdf5"
# elif args.dataset == "can":
#     target = f"/app/robomimic/datasets/can/can_feats{dataset_suffix}_w_cdm.hdf5"
#     source = f"/app/robomimic/datasets/can/can_demo.hdf5"
# elif args.dataset == "square":
#     target = f"/app/robomimic/datasets/square/square_feats{dataset_suffix}_w_cdm.hdf5"
#     source = f"/app/robomimic/datasets/square/square_demo.hdf5"
# else:
#     raise ValueError(f"Unknown dataset {args.dataset}. Please specify one of 'lift', 'can', or 'square'.")

if os.path.exists(source) and os.path.exists(target):
    sync_all_attributes(source, target)
else:
    print("Check your file paths!")

# Path to your dataset with divergence info
dataset_path = os.path.expanduser(target)

# Create Diffusion Policy configuration
config = config_factory(algo_name="diffusion_policy")

with config.values_unlocked():
    # Set dataset path
    config.train.data = dataset_path
    
    # Set output directory for results
    base_dir = args.output_dir
    if args.use_images:
        base_dir += "_images"
    elif args.end_to_end_image_training:
        base_dir += "_end2end_images"
    
    base_dir = os.path.join(base_dir, args.dataset)
    config.train.output_dir = base_dir

    # Search the output directory for existing experiments to set experiment name
    try:
        existing_exps = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("exp")]
        exp_nums = [int(d.split("exp")[-1]) for d in existing_exps if d.split("exp")[-1].isdigit()]
        exp_num = max(exp_nums) + 1 if exp_nums else 1
    except FileNotFoundError:
        exp_num = 1

    # config.experiment.name = f"exp{exp_num}"

    # Configure observation keys
    config.observation.modalities.obs.low_dim = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "object",
    ]
    if args.use_images:
        config.observation.modalities.obs.low_dim.append("robot0_eye_in_hand_feats")
        config.observation.modalities.obs.rgb = []

    elif args.end_to_end_image_training:
        config.observation.modalities.obs.rgb = ["robot0_eye_in_hand_image", "agentview_image"]
        config.observation.encoder.rgb.core_class = "VisualCore"
        config.observation.encoder.rgb.core_kwargs = {
            "backbone_class": "ResNet18Conv",
            "pool_class": "SpatialSoftmax",
            "feature_dimension": 512,
            "pretrained": False,
            "flatten": True,
        }

        config.observation.encoder.rgb.share = False

        config.observation.encoder.rgb.obs_randomizer_class = ["CropRandomizer", "ColorRandomizer"]
        config.observation.encoder.rgb.obs_randomizer_kwargs = [
            {"crop_height": 76, "crop_width": 76},
            {"brightness": 0.3, "contrast": 0.3, "saturation": 0.3, "hue": 0.1},
        ]
        config.observation.encoder.rgb.freeze = False
    
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
    config.train.num_epochs = args.epochs
    config.train.cuda = torch.cuda.is_available()
    
    # Save checkpoints
    config.experiment.save.enabled = True
    config.experiment.save.every_n_epochs = args.save_freq

    # Set experiment name with dataset portion, epochs, and save frequency
    config.experiment.name = f"{portion_prefix}_{args.epochs}_{args.save_freq}" #_exp{exp_num}"
    
    # Validation settings (disable to keep it simple for now)
    config.experiment.validate = False 
    if args.validate:
        config.experiment.rollout.enabled = True
        config.experiment.rollout.rate = args.save_freq
        config.experiment.rollout.n = 50  # number of rollouts per evaluation
        config.experiment.rollout.horizon = 400  # max steps per rollout
        
        # Enable rendering for both video and observations
        config.experiment.render = True
        config.experiment.render_video = True
        config.experiment.keep_all_videos = True # If False, only saves video for the best checkpoint
        
        # CRITICAL: When using images, configure environment to provide RGB observations during rollout
        if args.end_to_end_image_training:
            config.experiment.env_meta_update_dict = {
                "env_kwargs": {
                    "has_renderer": False,  # Set to False for offscreen rendering
                    "has_offscreen_renderer": True,  # Enable offscreen rendering for observations
                    "use_camera_obs": True,  # Enable camera observations
                    "camera_names": ["robot0_eye_in_hand", "agentview"],  # Match training cameras
                    "camera_heights": 84,
                    "camera_widths": 84,
                }
            }
    else:
        config.experiment.rollout.enabled = False


# Print config to verify
print("Training Configuration:")
print(config)

# Run training
train(config, device="cuda" if torch.cuda.is_available() else "cpu", resume=args.resume)
