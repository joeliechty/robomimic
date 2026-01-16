import robomimic
import os
import torch
from robomimic.config import config_factory
from robomimic.scripts.train import train

# Path to your dataset with divergence info
# Update this path if your file is located elsewhere
dataset_path = os.path.expanduser("/app/robomimic/datasets/lift/ph/low_dim_v15_w_cdm.hdf5")

# Create default BC configuration
config = config_factory(algo_name="bc")

with config.values_unlocked():
    # Set dataset path
    config.train.data = dataset_path
    
    # Set output directory for results
    # config.train.output_dir = os.path.expanduser("./exps/results/bc_divergence/vae")
    config.train.output_dir = os.path.expanduser("./exps/results/bc_no_divergence/vae")

    config.experiment.name = "bc_vae_test"

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
    # config.algo.loss.cdm_weight = 0.01
    config.algo.loss.cdm_weight = 0.0

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
