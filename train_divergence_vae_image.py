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

# add arguements for use_divergence_loss, div_loss_weight, dataset_path
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
        default=0.001,
        help="weight for divergence loss if used"
    )
    parser.add_argument(
        "--dataset", "-D",
        type=str,
        default="lift",
        help="dataset name: 'lift', 'can', or 'square'"
    )
    parser.add_argument(
        "--epochs","-E",
        type=int,
        default=200,
        help="number of training epochs"
    )
    parser.add_argument(
        "--output_dir", "-O",
        type=str,
        default="./exps/results/bc_rss/vae",
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
    target = f"/app/robomimic/datasets/lift/low_dim_v15{dataset_suffix}_w_cdm.hdf5"
elif args.dataset == "can":
    target = f"/app/robomimic/datasets/can/can_feats{dataset_suffix}_w_cdm.hdf5"
    source = f"/app/robomimic/datasets/can/can_demo.hdf5"
elif args.dataset == "square":
    target = f"/app/robomimic/datasets/square/square_feats{dataset_suffix}_w_cdm.hdf5"
    source = f"/app/robomimic/datasets/square/square_demo.hdf5"
else:
    raise ValueError(f"Unknown dataset {args.dataset}. Please specify one of 'lift', 'can', or 'square'.")

if args.dataset == "can" or args.dataset == "square":
    if os.path.exists(source) and os.path.exists(target):
        sync_all_attributes(source, target)
    else:
        print("Check your file paths!")

# Path to your dataset with divergence info
dataset_path = os.path.expanduser(target)

# Create default BC configuration
config = config_factory(algo_name="bc")

with config.values_unlocked():
    # Set dataset path
    config.train.data = dataset_path

    # Set output directory for results
    base_dir = args.output_dir
    if args.use_divergence_loss:
        base_dir += "_divergence"
        if args.use_images:
            base_dir += "_images"
        cdm_weight = args.div_loss_weight
        print(f"CDM Loss ENABLED with weight: {cdm_weight}, Images: {args.use_images} ")
    else:
        base_dir += "_no_divergence"
        if args.use_images:
            base_dir += "_images"
        cdm_weight = 0.0
        print(f"CDM Loss DISABLED (weight: {cdm_weight}), Images: {args.use_images} ")
    
    base_dir = os.path.join(base_dir, args.dataset)

    # Convert to absolute path to avoid issues with relative paths
    abs_base_dir = os.path.abspath(os.path.join("robomimic",base_dir))
    
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

    # config.experiment.name = f"exp{exp_num}"
    config.train.output_dir = base_dir

    # Configure observation keys
    # CRITICAL: 'robot0_eef_pos' and 'robot0_eef_quat' are required for 
    # the divergence computation (div_v_t) in the loss function.
    config.observation.modalities.obs.low_dim = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "object",
    ]
    if args.use_images:
        config.observation.modalities.obs.low_dim.append("robot0_eye_in_hand_feats")
        config.observation.modalities.obs.rgb = []
    # Disable RNN/Transformer to train VAE policy
    config.algo.rnn.enabled = False
    config.algo.transformer.enabled = False
    
    # Enable VAE policy
    config.algo.vae.enabled = True
    
    # VAE architecture settings
    config.algo.vae.latent_dim = 14  # Latent dimension (typically 2x action dim)
    config.algo.vae.latent_clip = None  # No latent clipping
    config.algo.vae.kl_weight = 1.0  # Beta-VAE weight for KL loss
    
    # VAE encoder/decoder network sizes
    config.algo.vae.encoder_layer_dims = (300, 400)
    config.algo.vae.decoder_layer_dims = (300, 400)
    
    # VAE decoder settings
    config.algo.vae.decoder.is_conditioned = True  # Condition decoder on observations
    config.algo.vae.decoder.reconstruction_sum_across_elements = False
    
    # VAE prior settings (standard N(0,1) prior)
    config.algo.vae.prior.learn = False  # Use standard Gaussian prior
    config.algo.vae.prior.is_conditioned = False
    config.algo.vae.prior.use_gmm = False
    config.algo.vae.prior_layer_dims = (300, 400)
    
    # NEW: Set divergence loss weight
    config.algo.loss.cdm_weight = cdm_weight

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
    config.experiment.rollout.enabled = False

# Print config to verify
print("Training Configuration:")
print(config)

# Run training
train(config, device="cuda" if torch.cuda.is_available() else "cpu")
