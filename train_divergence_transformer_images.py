import robomimic
import os
import torch
from robomimic.config import config_factory
from robomimic.scripts.train import train
import robomimic.algo.bc as bc
import argparse
import robomimic.utils.tensor_utils as TensorUtils
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
# source = "/app/robomimic/dataset/can/can_demo.hdf5"
# target = "/app/robomimic/dataset/can/can_feats_w_cdm.hdf5"

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
        "--output_dir", "-O",
        type=str,
        default="./exps/results/bc_rss/transformer",
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
    parser.add_argument(
        "--cdm_patience",
        type=int,
        default=10,
        help="number of epochs with no improvement before halving the CDM weight"
    )
    parser.add_argument(
        "--cdm_decay_factor",
        type=float,
        default=0.5,
        help="factor to multiply CDM weight by when a plateau is detected"
    )
    return parser.parse_args()

# --- Monkey-patch for observation history buffering during rollout ---
def get_action_with_history(self, obs_dict, goal_dict=None):
    assert not self.nets.training

    # Initialize buffer if needed
    if not hasattr(self, "obs_history"):
        self.obs_history = {}
        self.context_length = self.algo_config.transformer.context_length
    
    # Filter obs_dict to only include keys the policy expects
    # Use the observation keys from the observation encoder
    expected_keys = self.obs_shapes.keys()
    filtered_obs_dict = {k: v for k, v in obs_dict.items() if k in expected_keys}
        
    # Append current obs to history
    for k, v in filtered_obs_dict.items():
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

# --- Monkey-patch for CDM Weight Decay on Plateau ---
if args.use_divergence_loss:
    _original_train_on_batch = bc.BC_Transformer.train_on_batch
    _original_on_epoch_end = getattr(bc.BC_Transformer, "on_epoch_end", None)

    def train_on_batch_with_loss_tracking(self, batch, epoch, validate=False):
        info = _original_train_on_batch(self, batch, epoch, validate=validate)
        if not validate:
            if not hasattr(self, "_cdm_epoch_loss_sum"):
                self._cdm_epoch_loss_sum = 0.0
                self._cdm_epoch_batches = 0
            loss_val = info.get("Loss", 0.0)
            if isinstance(loss_val, torch.Tensor):
                loss_val = loss_val.item()
            self._cdm_epoch_loss_sum += loss_val
            self._cdm_epoch_batches += 1
        return info

    def on_epoch_end_with_cdm_decay(self, epoch):
        if _original_on_epoch_end is not None:
            _original_on_epoch_end(self, epoch)

        if not hasattr(self, "_cdm_epoch_loss_sum") or self._cdm_epoch_batches == 0:
            return

        avg_loss = self._cdm_epoch_loss_sum / self._cdm_epoch_batches
        self._cdm_epoch_loss_sum = 0.0
        self._cdm_epoch_batches = 0

        if not hasattr(self, "_cdm_best_loss"):
            self._cdm_best_loss = float("inf")
            self._cdm_patience_counter = 0

        if avg_loss < self._cdm_best_loss - 1e-6:
            self._cdm_best_loss = avg_loss
            self._cdm_patience_counter = 0
        else:
            self._cdm_patience_counter += 1
            print(f"[Epoch {epoch}] No loss improvement ({avg_loss:.6f} >= best {self._cdm_best_loss:.6f}). "
                  f"CDM plateau counter: {self._cdm_patience_counter}/{args.cdm_patience}")

        if self._cdm_patience_counter >= args.cdm_patience:
            current_weight = self.algo_config.loss.cdm_weight
            new_weight = current_weight * args.cdm_decay_factor
            if current_weight > 1e-10:
                print(f"[Epoch {epoch}] Plateau detected — decaying CDM weight: {current_weight:.2e} -> {new_weight:.2e}")
                with self.algo_config.values_unlocked():
                    self.algo_config.loss.cdm_weight = new_weight
                if hasattr(self, "cdm_weight"):
                    self.cdm_weight = new_weight
            self._cdm_patience_counter = 0  # reset after decay

    _original_log_info = bc.BC_Transformer.log_info

    def log_info_with_cdm_weight(self, info):
        log = _original_log_info(self, info)
        log["CDM_Weight"] = self.algo_config.loss.cdm_weight
        return log

    bc.BC_Transformer.train_on_batch = train_on_batch_with_loss_tracking
    bc.BC_Transformer.log_info = log_info_with_cdm_weight
    bc.BC_Transformer.on_epoch_end = on_epoch_end_with_cdm_decay
    print(f"Applied BC_Transformer monkey-patch for CDM weight decay on plateau "
          f"(patience={args.cdm_patience}, decay_factor={args.cdm_decay_factor})")
# -----------------------------------------------------

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
# if args.dataset == "lift":
#     target = f"datasets/lift/lift_feats{dataset_suffix}_w_cdm.hdf5"
#     source = f"datasets/lift/lift_demo.hdf5"
# elif args.dataset == "can":
#     target = f"datasets/can/can_feats{dataset_suffix}_w_cdm.hdf5"
#     source = f"datasets/can/can_demo.hdf5"
# elif args.dataset == "square":
#     target = f"datasets/square/square_feats{dataset_suffix}_w_cdm.hdf5"
#     source = f"datasets/square/square_demo.hdf5"
else:
    raise ValueError(f"Unknown dataset {args.dataset}. Please specify one of 'lift', 'can', or 'square'.")

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
    
    # Request divergence and score from dataset when using CDM
    if args.use_divergence_loss:
        config.train.dataset_keys = ["actions", "rewards", "dones", "divergence", "score"]
    
    # Set output directory for results
    base_dir = args.output_dir
    if args.use_divergence_loss:
        base_dir += "_divergence"
        if args.use_images:
            base_dir += "_images"
        elif args.end_to_end_image_training:
            base_dir += "_end2end_images"
        cdm_weight = args.div_loss_weight
        print(f"CDM Loss ENABLED with weight: {cdm_weight}, Images: {args.use_images} ")
    else:
        base_dir += "_no_divergence"
        if args.use_images:
            base_dir += "_images"
        elif args.end_to_end_image_training:
            base_dir += "_end2end_images"
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

        config.observation.encoder.rgb.obs_randomizer_class = "CropColorRandomizer"
        config.observation.encoder.rgb.obs_randomizer_kwargs = {
            "crop_height": 76,
            "crop_width": 76,
            "brightness": 0.3,
            "contrast": 0.3,
            "saturation": 0.3,
            "hue": 0.1,
        }
        config.observation.encoder.rgb.freeze = False
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
    config.train.batch_size = args.batch_size
    config.train.num_epochs = args.epochs
    config.train.seq_length = 10  # Must match transformer.context_length
    config.train.cuda = torch.cuda.is_available()
    
    # Save checkpoints
    config.experiment.save.enabled = True
    config.experiment.save.every_n_epochs = args.save_freq
    
    # Set experiment name with dataset portion, epochs, and save frequency
    config.experiment.name = f"{portion_prefix}_{args.epochs}_{args.save_freq}_{cdm_weight}" #_exp{exp_num}"
    
    # Rollout settings
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
