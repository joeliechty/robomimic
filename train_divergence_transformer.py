import robomimic
import os
import torch
from robomimic.config import config_factory
from robomimic.scripts.train import train
import robomimic.algo.bc as bc
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
    return parser.parse_args()

def get_action_with_history(self, obs_dict, goal_dict=None):
    assert not self.nets.training

    # Initialize buffer if needed
    if not hasattr(self, "obs_history"):
        self.obs_history = {}
        self.context_length = self.algo_config.transformer.context_length
        
    # Append current obs to history
    for k, v in obs_dict.items():
        if k not in self.obs_history:
            self.obs_history[k] = []
        self.obs_history[k].append(v)
        
    # Maintain context length
    for k, v_list in self.obs_history.items():
        while len(v_list) > self.context_length:
            v_list.pop(0)
            
    # Prepare input batch with correct shape [B, T, D]
    input_obs = {}
    for k, v_list in self.obs_history.items():
        # Stack along time dimension: [B, T, D]
        seq = torch.stack(v_list, dim=1) 
        
        # Pad if sequence is shorter than context_length
        current_len = seq.shape[1]
        if current_len < self.context_length:
            pad_len = self.context_length - current_len
            # Pad with the first observation
            first_obs = seq[:, 0:1, :]
            padding = first_obs.repeat(1, pad_len, *([1] * (first_obs.ndim - 2)))
            seq = torch.cat([padding, seq], dim=1)
            
        input_obs[k] = seq
        
    output = self.nets["policy"](input_obs, actions=None, goal_dict=goal_dict)

    if self.algo_config.transformer.supervise_all_steps:
        if self.algo_config.transformer.pred_future_acs:
            output = output[:, 0, :]
        else:
            output = output[:, -1, :]
    else:
        output = output[:, -1, :]

    return output

def reset_with_history(self):
    self.obs_history = {}

bc.BC_Transformer.get_action = get_action_with_history
bc.BC_Transformer.reset = reset_with_history

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
    base_dir = "./exps/results/bc_rss/transformer"
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
    
    # Enable Transformer architecture
    config.algo.rnn.enabled = False
    config.algo.transformer.enabled = True
    
    # Transformer architecture settings
    config.algo.transformer.context_length = 10  # Number of timesteps to condition on
    config.algo.transformer.embed_dim = 512
    config.algo.transformer.num_layers = 6
    config.algo.transformer.num_heads = 8
    config.algo.transformer.supervise_all_steps = False  # Only supervise last token
    
    # NEW: Set divergence loss weight
    config.algo.loss.cdm_weight = cdm_weight

    # Training settings
    config.train.batch_size = 256
    config.train.num_epochs = 200
    config.train.seq_length = 10  # Must match transformer.context_length
    config.train.cuda = torch.cuda.is_available()
    
    # Save checkpoints
    config.experiment.save.enabled = True
    config.experiment.save.every_n_epochs = 1
    
    # Validation settings (disable to keep it simple for now)
    config.experiment.validate = False 

# Print config to verify
print("Training Configuration:")
print(config)

# Run training
train(config, device="cuda" if torch.cuda.is_available() else "cpu")
