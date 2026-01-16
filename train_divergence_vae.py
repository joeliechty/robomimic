import robomimic
import os
import torch
from robomimic.config import config_factory
from robomimic.scripts.train import train
import argparse

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
        default=0.01,
        help="weight for divergence loss if used"
    )
    parser.add_argument(
        "--dataset_path","-D",
        type=str,
        default="/app/robomimic/datasets/lift/ph/low_dim_v15_w_cdm.hdf5",
        help="path to dataset hdf5 file"
    )
    parser.add_argument(
        "--epochs","-E",
        type=int,
        default=200,
        help="number of training epochs"
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
    base_dir = "./exps/results/bc_rss/vae"
    if args.use_divergence_loss:
        base_dir += "_divergence"
        cdm_weight = args.div_loss_weight
        print(f"CDM Loss ENABLED with weight: {cdm_weight}")
    else:
        base_dir += "_no_divergence"
        cdm_weight = 0.0
        print(f"CDM Loss DISABLED (weight: {cdm_weight})")

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

    config.experiment.name = f"exp{exp_num}"
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
    config.train.batch_size = 256
    config.train.num_epochs = args.epochs
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
