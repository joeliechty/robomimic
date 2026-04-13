import robomimic
import os
import sys
import shutil
import tempfile
import atexit
import signal
import getpass
import torch
from robomimic.config import config_factory
from robomimic.scripts.train import train
import robomimic.algo.bc as bc
import argparse
import robomimic.utils.tensor_utils as TensorUtils
import h5py
import numpy as np

def sync_all_attributes(source_path, target_path):
    """Sync HDF5 attributes from the full demo dataset into the feats dataset.

    Always reads from the full demo source so that half/quarter feats files
    (which are subsets of the full dataset) can look up each of their demos
    in the source by the same demo key.
    """
    print(f"Syncing attributes from {source_path} to {target_path}...")

    with h5py.File(source_path, 'r') as f_src, h5py.File(target_path, 'a') as f_tgt:
        # 1. Sync global /data attributes (env_args, etc.)
        if "data" in f_src and "data" in f_tgt:
            for k, v in f_src["data"].attrs.items():
                f_tgt["data"].attrs[k] = v
            print("  [OK] Global 'data' attributes synced.")

        # 2. Sync per-demo attributes (num_samples, model_file, etc.)
        # Iterate over TARGET demos so half/quarter subsets only process the
        # demos they actually contain, fetching each from the full source.
        target_demos = [k for k in f_tgt["data"].keys() if k.startswith("demo_")]
        synced = 0
        missing_in_source = 0
        for demo in target_demos:
            if demo in f_src["data"]:
                for k, v in f_src[f"data/{demo}"].attrs.items():
                    f_tgt[f"data/{demo}"].attrs[k] = v
                synced += 1
            else:
                print(f"  [Warning] {demo} found in target but not in source. Skipping.")
                missing_in_source += 1

        print(f"  [OK] Attributes synced for {synced}/{len(target_demos)} demos."
              + (f" ({missing_in_source} missing in source)" if missing_in_source else ""))

        # 3. Rebuild /mask filter keys to only include demos present in the target.
        # The full demo file's mask/train and mask/valid list all 200 demos, but
        # half/quarter feats files only contain a subset. Prune stale entries so
        # load_demo_info doesn't try to open non-existent demos.
        if "mask" in f_src:
            target_demo_set = set(target_demos)
            for key_name in f_src["mask"].keys():
                src_demos_in_mask = [d.decode("utf-8") if isinstance(d, bytes) else d
                                     for d in f_src[f"mask/{key_name}"][:]]
                filtered = [d for d in src_demos_in_mask if d in target_demo_set]
                if f"mask/{key_name}" in f_tgt:
                    del f_tgt[f"mask/{key_name}"]
                if "mask" not in f_tgt:
                    f_tgt.create_group("mask")
                f_tgt.create_dataset(f"mask/{key_name}",
                                     data=np.array(filtered, dtype="S"))
            print(f"  [OK] Filter keys rebuilt: {list(f_src['mask'].keys())} "
                  f"(filtered to {len(target_demo_set)} target demos)")


# --- Per-job local HDF5 copy (parallel-safe) + cleanup ---
_LOCAL_DATASET_COPY_PATH = None
_KEEP_LOCAL_COPY_FLAG = False
_LOCAL_COPY_CLEANUP_DONE = False


def _remove_local_dataset_copy():
    """Remove the temporary per-job dataset file if registered and not --keep_local_copy."""
    global _LOCAL_COPY_CLEANUP_DONE
    if _KEEP_LOCAL_COPY_FLAG or _LOCAL_COPY_CLEANUP_DONE:
        return
    p = _LOCAL_DATASET_COPY_PATH
    if p and os.path.isfile(p):
        try:
            os.remove(p)
            print(f"Removed temporary dataset copy: {p}")
        except OSError as e:
            print(f"Warning: could not remove temporary dataset copy {p}: {e}")
    _LOCAL_COPY_CLEANUP_DONE = True


def register_local_dataset_cleanup(local_path, keep):
    """
    Register deletion of local_path on exit or SIGTERM/SIGINT unless keep is True.
    """
    global _LOCAL_DATASET_COPY_PATH, _KEEP_LOCAL_COPY_FLAG
    _LOCAL_DATASET_COPY_PATH = local_path
    _KEEP_LOCAL_COPY_FLAG = keep
    if keep:
        print(f"Keeping local dataset copy (--keep_local_copy): {local_path}")
        return
    atexit.register(_remove_local_dataset_copy)

    def _signal_cleanup(signum, frame):
        _remove_local_dataset_copy()
        sys.exit(128 + signum)

    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _signal_cleanup)
        except ValueError:
            # e.g. non-main thread
            pass


def get_local_copy_base_dir(args):
    """Resolve directory for per-job dataset copies (plan: SLURM_TMPDIR, then scratch/local, else temp)."""
    if getattr(args, "local_copy_dir", None):
        return os.path.abspath(os.path.expanduser(args.local_copy_dir))
    slurm_tmp = os.environ.get("SLURM_TMPDIR")
    if slurm_tmp:
        return os.path.abspath(slurm_tmp)
    user = os.environ.get("USER") or getpass.getuser()
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    scratch_local = os.path.join("/scratch/local", user, job_id)
    if os.path.isdir("/scratch/local"):
        return scratch_local
    return tempfile.gettempdir()


def make_unique_local_dataset_path(shared_target, copy_dir):
    """Unique filename under copy_dir to avoid collisions across array tasks / PIDs."""
    base = os.path.basename(shared_target)
    stem, ext = os.path.splitext(base)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    pid = os.getpid()
    uniq = f"{stem}_job{job_id}_pid{pid}{ext}"
    return os.path.join(copy_dir, uniq)


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
        "--min_cdm_weight",
        type=float,
        default=1e-7,
        help="minimum CDM weight at the end of the decay phase"
    )
    parser.add_argument(
        "--seed", "-S",
        type=int,
        default=None,
        help="random seed for reproducibility (omit to leave unseeded)"
    )
    parser.add_argument(
        "--action_chunk_size", "-ACS",
        type=int,
        default=1,
        help="number of future actions to predict per timestep (1 = no chunking)"
    )
    parser.add_argument(
        "--use_local_dataset_copy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy feats HDF5 to job-local storage so parallel jobs do not contend on one file (default: on). "
        "Use --no-use_local_dataset_copy to read the shared path in place.",
    )
    parser.add_argument(
        "--local_copy_dir",
        type=str,
        default=None,
        help="Directory for the temporary dataset copy (default: SLURM_TMPDIR, else /scratch/local/$USER/$SLURM_JOB_ID, else system temp).",
    )
    parser.add_argument(
        "--keep_local_copy",
        action="store_true",
        help="Do not delete the temporary dataset copy after training (debugging).",
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

# --- Monkey-patch for BC_Transformer_Chunking with observation history buffering ---
from collections import deque as collections_deque

def get_action_with_history_chunking(self, obs_dict, goal_dict=None):
    """
    Get action for BC_Transformer_Chunking with temporal ensembling.
    Runs inference every step, keeps a buffer of last 16 action chunks,
    and returns the weighted average of aligned predictions.
    """
    assert not self.nets.training

    # Initialize buffers if needed
    if not hasattr(self, "obs_history"):
        self.obs_history = {}
        self.context_length = self.algo_config.transformer.context_length
        self.action_chunk_size = self.algo_config.transformer.action_chunk_size
        # Buffer to store action chunks: list of [1, chunk_size, ac_dim] tensors
        # self.action_chunk_buffer = collections_deque(maxlen=16)
        self.action_chunk_buffer = collections_deque(maxlen=self.action_chunk_size)
        # Track timestep for temporal alignment
        self.timestep = 0

    # Filter obs_dict to only include keys the policy expects
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

    # Run inference every step
    output = self.nets["policy"](input_obs, actions=None, goal_dict=goal_dict)
    # output shape: [1, T, chunk_size, ac_dim]
    # Take the last timestep's chunk
    action_chunk = output[:, -1, :, :]  # [1, chunk_size, ac_dim]

    # Add the new action chunk to buffer with current timestep
    self.action_chunk_buffer.append((self.timestep, action_chunk))

    # Temporal ensembling: collect all predictions for current timestep
    predictions = []
    weights = []

    for pred_timestep, chunk in self.action_chunk_buffer:
        # Calculate the offset: how many steps ahead was this prediction made?
        offset = self.timestep - pred_timestep

        # If offset is within the chunk size, this chunk has a prediction for current timestep
        if 0 <= offset < self.action_chunk_size:
            action_for_current_step = chunk[:, offset, :]  # [1, ac_dim]
            predictions.append(action_for_current_step)

            # Exponential weighting: more recent predictions get higher weight
            # You can adjust the decay factor (0.9) to control the weighting
            weight = 0.9 ** offset
            weights.append(weight)

    # Normalize weights
    if len(predictions) > 0:
        predictions = torch.stack(predictions, dim=0)  # [num_predictions, 1, ac_dim]
        weights = torch.tensor(weights, device=predictions.device, dtype=predictions.dtype)
        weights = weights / weights.sum()  # Normalize to sum to 1

        # Weighted average: [num_predictions, 1, ac_dim] * [num_predictions, 1, 1] -> [1, ac_dim]
        action = (predictions * weights.view(-1, 1, 1)).sum(dim=0)
    else:
        # Fallback: if no predictions available (shouldn't happen), use first action from chunk
        action = action_chunk[:, 0, :]

    # Increment timestep counter
    self.timestep += 1

    return action

def reset_with_history_chunking(self):
    """Reset observation history and temporal ensembling buffers."""
    self.obs_history = {}
    self.action_chunk_buffer = collections_deque(maxlen=16)
    self.timestep = 0

bc.BC_Transformer_Chunking.get_action = get_action_with_history_chunking
bc.BC_Transformer_Chunking.reset = reset_with_history_chunking
print("Applied BC_Transformer_Chunking monkey-patch for temporal ensembling with action chunk buffer during rollout")

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


# -- Monkey-patch for memory-efficient data loading with images during training ---
import robomimic.utils.dataset as ds

def get_item_efficient(self, index):
    """
    Monkey-patched get_item to only fetch `context_length` images, 
    but the full chunked sequence for actions/goals/divergence data.
    """
    demo_id = self._index_to_demo_id[index]
    demo_start_index = self._demo_id_to_start_indices[demo_id]
    demo_length = self._demo_id_to_demo_length[demo_id]

    demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
    index_in_demo = index - demo_start_index + demo_index_offset

    demo_length_offset = 0 if self.pad_seq_length else (self.seq_length - 1)
    end_index_in_demo = demo_length - demo_length_offset

    # FETCH FULL SEQUENCE for actions/divergence (fast, low memory)
    meta = self.get_dataset_sequence_from_demo(
        demo_id,
        index_in_demo=index_in_demo,
        keys=self.dataset_keys,
        num_frames_to_stack=self.n_frame_stack - 1,
        seq_length=self.seq_length
    )

    goal_index = None
    if self.goal_mode == "last":
        goal_index = end_index_in_demo - 1

    # DETERMINE HOW MANY IMAGES WE ACTUALLY NEED
    # seq_length = context_length + action_chunk_size - 1
    # We retrieve the global args to find the action_chunk_size
    import __main__
    chunk_size = __main__.args.action_chunk_size if hasattr(__main__, 'args') else 1
    obs_seq_length = self.seq_length - chunk_size + 1

    # FETCH SHORT SEQUENCE for images (saves massive amounts of memory)
    meta["obs"] = self.get_obs_sequence_from_demo(
        demo_id,
        index_in_demo=index_in_demo,
        keys=self.obs_keys,
        num_frames_to_stack=self.n_frame_stack - 1,
        seq_length=obs_seq_length,
        prefix="obs"
    )

    if self.goal_mode is not None:
        goal = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=goal_index,
            keys=self.obs_keys,
            num_frames_to_stack=0,
            seq_length=1,
            prefix="next_obs",
        )
        meta["goal_obs"] = {k: goal[k][0] for k in goal}

    return meta

# Apply the patch
ds.SequenceDataset.get_item = get_item_efficient
print("Applied SequenceDataset monkey-patch for memory-efficient Chunking DataLoad")

bc.BC_Transformer.process_batch_for_training = process_batch_for_training_with_divergence
print("Applied BC_Transformer monkey-patch for process_batch_for_training (CDM)")
# ----------------------------------------

args = parse_args()

# --- Monkey-patch for CDM Weight Decay Schedule ---
if args.use_divergence_loss:
    _original_log_info = bc.BC_Transformer.log_info

    def log_info_with_cdm_weight(self, info):
        log = _original_log_info(self, info)
        log["CDM_Weight"] = self.algo_config.loss.cdm_weight
        return log

    bc.BC_Transformer.log_info = log_info_with_cdm_weight

    import math
    _max_cdm_weight = args.div_loss_weight
    _min_cdm_weight = args.min_cdm_weight
    _total_epochs = args.epochs
    _ramp_end = 0.1 * _total_epochs      # 10% mark
    _decay_start = 0.8 * _total_epochs    # 80% mark

    def _compute_cdm_weight(epoch):
        if epoch <= _ramp_end:
            # Linear ramp from 0 to max over first 10%
            return _max_cdm_weight * (epoch / _ramp_end) if _ramp_end > 0 else _max_cdm_weight
        elif epoch <= _decay_start:
            # Hold at max from 10% to 80%
            return _max_cdm_weight
        else:
            # Cosine decay from max to min over 80% to 100%
            decay_progress = (epoch - _decay_start) / (_total_epochs - _decay_start)
            return _min_cdm_weight + 0.5 * (_max_cdm_weight - _min_cdm_weight) * (
                1 + math.cos(math.pi * decay_progress)
            )

    # Weight is set at the start of each epoch's first batch so it is correct
    # for every epoch — including when resuming (algo_config is not checkpointed).
    _sched_last_epoch = [-1]
    _original_train_on_batch_sched = bc.BC_Transformer.train_on_batch

    def train_on_batch_with_cdm_schedule(self, batch, epoch, validate=False):
        if epoch != _sched_last_epoch[0]:
            _sched_last_epoch[0] = epoch
            new_weight = _compute_cdm_weight(epoch)
            with self.algo_config.values_unlocked():
                self.algo_config.loss.cdm_weight = new_weight
            if hasattr(self, "cdm_weight"):
                self.cdm_weight = new_weight
            phase = "ramp" if epoch <= _ramp_end else ("hold" if epoch <= _decay_start else "decay")
            print(f"[Epoch {epoch}] CDM weight: {new_weight:.2e} ({phase})")
        return _original_train_on_batch_sched(self, batch, epoch, validate=validate)

    bc.BC_Transformer.train_on_batch = train_on_batch_with_cdm_schedule
    print(f"Applied CDM weight schedule: ramp 0→{_max_cdm_weight:.2e} (epochs 0-{_ramp_end:.0f}), "
          f"hold (epochs {_ramp_end:.0f}-{_decay_start:.0f}), "
          f"cosine decay→{_min_cdm_weight:.2e} (epochs {_decay_start:.0f}-{_total_epochs})")
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


if args.dataset in ["lift", "can", "square", "tool"]:
    target = f"datasets/{args.dataset}/{args.dataset}_feats{dataset_suffix}_w_cdm.hdf5"
    source = f"datasets/{args.dataset}/{args.dataset}_demo.hdf5"
else:
    raise ValueError(f"Unknown dataset {args.dataset}. Please specify one of 'lift', 'can', 'square', or 'tool'.")

target = os.path.abspath(os.path.expanduser(target))
source = os.path.abspath(os.path.expanduser(source))

# Path to your dataset with divergence info (per-job copy when enabled)
dataset_path = target
if os.path.exists(source) and os.path.exists(target):
    if args.use_local_dataset_copy:
        copy_dir = get_local_copy_base_dir(args)
        os.makedirs(copy_dir, exist_ok=True)
        local_target = make_unique_local_dataset_path(target, copy_dir)
        print(
            f"Per-job dataset copy (parallel-safe):\n"
            f"  shared: {target}\n"
            f"  local:  {local_target}"
        )
        try:
            shutil.copy2(target, local_target)
        except OSError as e:
            raise RuntimeError(
                f"Failed to copy dataset to {local_target}. "
                f"Check space and permissions in {copy_dir}."
            ) from e
        dataset_path = local_target
        register_local_dataset_cleanup(local_target, args.keep_local_copy)
        if not args.resume:
            sync_all_attributes(source, local_target)
        else:
            print("Resuming training — skipping attribute sync (already done on initial run).")
    else:
        if not args.resume:
            sync_all_attributes(source, target)
        else:
            print("Resuming training — skipping attribute sync (already done on initial run).")
else:
    print("Check your file paths!")

dataset_path = os.path.abspath(os.path.expanduser(dataset_path))

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

        config.observation.encoder.rgb.obs_randomizer_class = ["CropRandomizer", "ColorRandomizer"]
        config.observation.encoder.rgb.obs_randomizer_kwargs = [
            {"crop_height": 76, "crop_width": 76},
            {"brightness": 0.3, "contrast": 0.3, "saturation": 0.3, "hue": 0.1},
        ]
        config.observation.encoder.rgb.freeze = False
    # Enable Transformer architecture
    config.algo.rnn.enabled = False
    config.algo.transformer.enabled = True
    
    # Transformer architecture settings — use a smaller model for lift to reduce overfitting
    if args.dataset == "lift":
        context_length = 2
        config.algo.transformer.context_length = context_length
        config.algo.transformer.embed_dim = 128
        config.algo.transformer.num_layers = 3
        config.algo.transformer.num_heads = 4
    else:
        context_length = 3
        config.algo.transformer.context_length = context_length
        config.algo.transformer.embed_dim = 512
        config.algo.transformer.num_layers = 6
        config.algo.transformer.num_heads = 8
    config.algo.transformer.supervise_all_steps = False  # Only supervise last token

    # Action chunking settings
    config.algo.transformer.action_chunk_size = args.action_chunk_size

    # NEW: Set divergence loss weight
    config.algo.loss.cdm_weight = cdm_weight

    if args.dataset in ["tool"]:
        config.algo.loss.l2_weight = 0.0  # Turn off L2 loss
        config.algo.loss.l1_weight = 1.0  # Turn on L1 loss

    # Training settings
    config.train.batch_size = args.batch_size
    config.train.num_epochs = args.epochs
    # seq_length must account for action chunking: need context_length + action_chunk_size - 1 timesteps
    config.train.seq_length = context_length + args.action_chunk_size - 1
    config.train.cuda = torch.cuda.is_available()
    
    # Save checkpoints
    config.experiment.save.enabled = True
    config.experiment.save.every_n_epochs = args.save_freq
    
    # Set experiment name with dataset portion, epochs, and save frequency
    if args.use_divergence_loss:
        decay_type = f"ramp_hold_cosine_max{cdm_weight}_min{args.min_cdm_weight}"
    else:
        decay_type = "no_cdm"

    # Set random seed
    if args.seed is not None:
        config.train.seed = args.seed

    seed_suffix = f"_seed{args.seed}" if args.seed is not None else ""
    chunk_suffix = f"_chunk{args.action_chunk_size}" if args.action_chunk_size > 1 else ""
    config.experiment.name = f"{portion_prefix}_{args.epochs}_{args.save_freq}_{decay_type}{seed_suffix}{chunk_suffix}" #_exp{exp_num}"
    
    # Rollout settings
    if args.validate:
        config.experiment.rollout.enabled = True
        config.experiment.rollout.rate = args.save_freq
        config.experiment.rollout.n = 50  # number of rollouts per evaluation
        if args.dataset == "tool":
            config.experiment.rollout.horizon = 800  # tool task is longer, so use longer horizon
        else:
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

    # Tell the dataloader to use the 'train' split for training
    config.train.hdf5_filter_key = "train"
    
    # Enable validation and tell it to use the 'valid' split
    config.experiment.validate = True
    config.train.hdf5_validation_filter_key = "valid"

# Print config to verify
print("Training Configuration:")
print(config)

# Run training
train(config, device="cuda" if torch.cuda.is_available() else "cpu", resume=args.resume)
