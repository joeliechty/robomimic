import robomimic
import os
import torch
from robomimic.config import config_factory
from robomimic.scripts.train import train
import robomimic.algo.bc as bc
import argparse
import robomimic.utils.tensor_utils as TensorUtils

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
        default=0.0001,
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

# --- Monkey-patch for observation history buffering during rollout ---
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
print("Applied BC_Transformer monkey-patch for observation history buffering during rollout")

# --- Monkey-patch for CDM Support ---
def process_batch_for_training_with_divergence(self, batch):
    input_batch = dict()
    h = self.context_length
    input_batch["obs"] = {k: batch["obs"][k][:, :h, :] for k in batch["obs"]}
    input_batch["goal_obs"] = batch.get("goal_obs", None)  # goals may not be present

    if self.supervise_all_steps:
        # supervision on entire sequence (instead of just current timestep)
        if self.pred_future_acs:
            ac_start = h - 1
        else:
            ac_start = 0
        input_batch["actions"] = batch["actions"][:, ac_start:ac_start+h, :]
    else:
        # just use current timestep
        input_batch["actions"] = batch["actions"][:, h-1, :]

    if self.pred_future_acs:
        assert input_batch["actions"].shape[1] == h

    # --- CDM related changes ---
    # Extract divergence and score if present in batch
    # They should have shape [B, T] or [B, T, D] coming from dataloader because of seq_length
    if "divergence" in batch:
        if self.supervise_all_steps:
            if self.pred_future_acs:
                ac_start = h - 1
            else:
                ac_start = 0
            input_batch["divergence"] = batch["divergence"][:, ac_start:ac_start+h]
        else:
            input_batch["divergence"] = batch["divergence"][:, h-1]
    
    if "score" in batch:
        if self.supervise_all_steps:
            if self.pred_future_acs:
                ac_start = h - 1
            else:
                ac_start = 0
            input_batch["score"] = batch["score"][:, ac_start:ac_start+h, :]
        else:
            input_batch["score"] = batch["score"][:, h-1, :]
    # ---------------------------

    input_batch = TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
    return input_batch

bc.BC_Transformer.process_batch_for_training = process_batch_for_training_with_divergence
print("Applied BC_Transformer monkey-patch for process_batch_for_training (CDM)")
# ----------------------------------------

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
    args.dataset_path = f"/app/robomimic/datasets/lift/low_dim_v15{dataset_suffix}_w_cdm.hdf5"
elif args.dataset == "can":
    args.dataset_path = f"/app/robomimic/datasets/can/can_feats{dataset_suffix}_w_cdm.hdf5"
elif args.dataset == "square":
    args.dataset_path = f"/app/robomimic/datasets/square/square_feats{dataset_suffix}_w_cdm.hdf5"
else:
    raise ValueError(f"Unknown dataset {args.dataset}. Please specify one of 'lift', 'can', or 'square'.")

# Path to your dataset with divergence info
dataset_path = os.path.expanduser(args.dataset_path)

# Create default BC configuration
config = config_factory(algo_name="bc")

with config.values_unlocked():
    # Set dataset path
    config.train.data = dataset_path
    
    # Request divergence and score from dataset when using CDM
    if args.use_divergence_loss:
        config.train.dataset_keys = ["actions", "rewards", "dones", "divergence", "score"]
    
    # Set output directory for results
    base_dir = "./exps/results/bc_rss/transformer"
    if args.use_divergence_loss:
        base_dir += "_divergence"
        cdm_weight = args.div_loss_weight
        print(f"CDM Loss ENABLED with weight: {cdm_weight}")
    else:
        base_dir += "_no_divergence"
        cdm_weight = 0.0
        print(f"CDM Loss DISABLED (weight: {cdm_weight})")
    
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
    config.train.num_epochs = args.epochs
    config.train.seq_length = 10  # Must match transformer.context_length
    config.train.cuda = torch.cuda.is_available()
    
    # Save checkpoints
    config.experiment.save.enabled = True
    config.experiment.save.every_n_epochs = args.save_freq
    
    # Set experiment name with dataset portion, epochs, and save frequency
    config.experiment.name = f"{portion_prefix}_{args.epochs}_{args.save_freq}" #_exp{exp_num}"
    
    # Validation settings (disable to keep it simple for now)
    config.experiment.validate = False 

    # Rollout settings
    config.experiment.rollout.enabled = False
    
# Print config to verify
print("Training Configuration:")
print(config)

# Run training
train(config, device="cuda" if torch.cuda.is_available() else "cpu")
