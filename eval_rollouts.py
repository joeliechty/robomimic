#!/usr/bin/env python
"""
Convenient script for evaluating trained models from the experiments.

Usage:
    python3 eval_rollouts.py -M transformer -T lift -DS F -TE 500 -SF 20 -EE 160 -V -SD
This will evaluate the transformer model trained on the full lift dataset for 500 epochs,
saving the video and data for epoch 160, using the default 50 rollouts and seed 0.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from tqdm import tqdm
import json

# Import the run_trained_agent function directly
from robomimic.scripts.run_trained_agent import run_trained_agent
import robomimic.algo.bc as bc
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils

# --- Logging utility to save console output to file ---
class Logger:
    """Tee-like class that writes to both stdout and a file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

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

# Apply the monkey patch for BC_Transformer
bc.BC_Transformer.get_action = get_action_with_history
bc.BC_Transformer.reset = reset_with_history
print("Applied BC_Transformer monkey-patch for observation history buffering during rollout")

def fix_checkpoint_camera_names(model_path):
    """Fix camera_names format in env_meta if it's stored as a string instead of list.
    
    Args:
        model_path: Path to the model checkpoint file
    """
    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(model_path)
    
    # Fix camera_names if it's a string
    if "env_metadata" in ckpt_dict:
        env_meta = ckpt_dict["env_metadata"]
        if "env_kwargs" in env_meta and "camera_names" in env_meta["env_kwargs"]:
            camera_names = env_meta["env_kwargs"]["camera_names"]
            if isinstance(camera_names, str):
                print(f"Fixed camera_names in env_meta: '{camera_names}' -> ['{camera_names}']")
                env_meta["env_kwargs"]["camera_names"] = [camera_names]
                # Save the checkpoint with the fix
                torch.save(ckpt_dict, model_path)
                return True  # Indicate that a fix was applied
    
    return False  # No fix needed

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained robomimic models")
    parser.add_argument("--model", "-M", type=str, required=True, choices=["transformer", "mlp", "diffusion", "diffusion_policy", "vae"], help="Model type: transformer, vae, diffusion, or diffusion_policy")
    parser.add_argument("--task", "-T", type=str, required=True, choices=["lift", "can", "square"], help="Task name")
    parser.add_argument("--divergence", "-CDM", action="store_true", help="Use divergence model")  
    parser.add_argument("--images", "-I", action="store_true", help="Model trained with images")
    parser.add_argument("--dataset_size", "-DS", type=str, required=True, choices=["F", "H1", "H2", "Q1", "Q2", "Q3", "Q4"], help="Dataset size: F (full), H1/H2 (half), Q1-Q4 (quarter)")
    parser.add_argument("--training_epochs", "-TE", type=int, required=True, help="Number of training epochs (e.g., 500, 1000)") 
    parser.add_argument("--save_freq", "-SF", type=int, required=True, help="Model save frequency during training (e.g., 5, 20)")
    parser.add_argument("--eval_epoch", "-EE", type=int, default=None, help="Specific epoch to evaluate. If not provided, will use 'last.pth'")
    parser.add_argument("--n_rollouts", "-ROLL", type=int, default=50, help="Number of evaluation rollouts (default: 50)")
    parser.add_argument("--seed", "-S", type=int, default=0, help="Random seed for evaluation (default: 0)")
    parser.add_argument("--video", "-V", action="store_true", help="Save evaluation video")
    parser.add_argument("--save_data", "-SD", action="store_true", help="Save rollout data (.hdf5) and stats (.json)")
    parser.add_argument("--horizon", "-H", type=int, default=400, help="Maximum horizon for rollouts (default: 400)")
    parser.add_argument("--camera_names", "-CAMS", type=str, nargs='+', default=["agentview"], help="Camera names for video rendering")
    # Loop mode arguments
    parser.add_argument("--loop", "-LOOP", action="store_true", help="Loop through all epochs from save_freq to training_epochs")
    parser.add_argument("--start_epoch", "-START", type=int, default=None, help="Starting epoch for loop mode (default: save_freq)")
    parser.add_argument("--end_epoch", "-END", type=int, default=None, help="Ending epoch for loop mode (default: training_epochs)")
    parser.add_argument("--eval_freq", "-EF", type=int, default=None, help="Evaluation frequency for loop mode (default: same as save_freq). Must be a multiple of save_freq.")
    # legacy experiment number argument
    parser.add_argument("--exp", type=int, default=None, help="[Legacy] Experiment number for old naming convention")
    return parser.parse_args()

def find_model_path(model_type, divergence, images, task, dataset_size, training_epochs, save_freq, epoch=None, exp_num=None):
    """Find the path to the trained model checkpoint.
    
    Args:
        model_type: transformer, vae, diffusion, or diffusion_policy
        divergence: whether model uses divergence
        images: whether model is trained with images
        task: lift, can, or square
        dataset_size: F, H1, H2, Q1-Q4
        training_epochs: number of training epochs (e.g., 500, 1000)
        save_freq: model save frequency (e.g., 5, 20)
        epoch: specific epoch to evaluate
        exp_num: [Legacy] experiment number for old naming convention

    Returns:
        model_path: path to the model checkpoint file
    """
    
    # Handle legacy exp_num format for backward compatibility
    if exp_num is not None:
        # Old format: exp{exp_num}
        if model_type == "diffusion":
            base_dir = "robomimic/exps/results/bc_rss/diffusion_policy"
        else:
            if model_type == "mlp":
                base_dir = "robomimic/exps/results/bc_rss/mlp"
            elif model_type == "transformer":
                base_dir = "robomimic/exps/results/bc_rss/transformer"
            elif model_type == "vae":
                base_dir = "robomimic/exps/results/bc_rss/vae"
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            if divergence:
                base_dir += "_divergence"
            else:
                base_dir += "_no_divergence"
        
        exp_dir = os.path.join(base_dir, task, f"exp{exp_num}")
    else:
        # New format: {dataset_size}_{training_epochs}_{save_freq}
        # Determine base directory based on model type
        if model_type in ["diffusion", "diffusion_policy"]:
            model_dir_name = "diffusion_policy"
        elif model_type == "mlp":
            model_dir_name = "mlp"
        elif model_type == "transformer":
            model_dir_name = "transformer"
        elif model_type == "vae":
            model_dir_name = "vae"
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Add divergence suffix
        if model_dir_name != "diffusion_policy":
            if divergence:
                model_dir_name += "_divergence"
            else:
                model_dir_name += "_no_divergence"
        
        # Add images suffix if applicable
        if images:
            model_dir_name += "_images"
        
        base_dir = f"robomimic/exps/results/bc_rss/{model_dir_name}"
        exp_folder = f"{dataset_size}_{training_epochs}_{save_freq}"
        exp_dir = os.path.join(base_dir, task, exp_folder)
    
    if not os.path.exists(exp_dir):
        print(f"Error: Experiment directory not found: {exp_dir}")
        sys.exit(1)
    
    # Find the timestamp subdirectory (there should be one)
    subdirs = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
    if len(subdirs) == 0:
        print(f"Error: No timestamp subdirectory found in {exp_dir}")
        sys.exit(1)
    
    # Use the most recent timestamp if multiple exist
    timestamp_dir = os.path.join(exp_dir, sorted(subdirs)[-1])
    
    # Find model checkpoint
    if epoch is None:
        raise NotImplementedError("Evaluation without specifying epoch is not implemented.")        
    else:
        # Find specific epoch checkpoint
        models_dir = os.path.join(timestamp_dir, "models")
        if not os.path.exists(models_dir):
            print(f"Error: Models directory not found: {models_dir}")
            sys.exit(1)
        
        # Look for checkpoint files matching the epoch
        model_files = [f for f in os.listdir(models_dir) if f.startswith(f"model_epoch_{epoch}") and f.endswith(".pth")]
        
        if len(model_files) == 0:
            print(f"Error: No model found for epoch {epoch} in {models_dir}")
            print(f"Available model files:")
            for f in sorted(os.listdir(models_dir)):
                if f.endswith(".pth"):
                    print(f"  {f}")
            sys.exit(1)
        
        # Use the first matching file (there might be multiple with success rates in name)
        model_path = os.path.join(models_dir, model_files[0])
    
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found: {model_path}")
        sys.exit(1)
    
    return model_path

def eval_single_model(args):
    """Evaluate a single model checkpoint.
    
    Args:
        args: argparse.Namespace with evaluation arguments
    """
    # Find model checkpoint
    if args.exp is not None:
        print(f"Looking for {args.model} model for task '{args.task}', experiment {args.exp}...")
        model_path = find_model_path(
            args.model, args.divergence, args.images, args.task, 
            None, None, None, args.eval_epoch, exp_num=args.exp
        )
    else:
        divergence_str = "with divergence" if args.divergence else "without divergence"
        images_str = "with images" if args.images else ""
        print(f"Looking for {args.model} model {divergence_str} {images_str} for task '{args.task}'...")
        print(f"  Dataset: {args.dataset_size}, Training epochs: {args.training_epochs}, Save freq: {args.save_freq}")
        model_path = find_model_path(
            args.model, args.divergence, args.images, args.task,
            args.dataset_size, args.training_epochs, args.save_freq, args.eval_epoch
        )
    print(f"Found model: {model_path}")
    
    # Build arguments for run_trained_agent
    epoch_str = f"epoch{args.eval_epoch}" if args.eval_epoch is not None else "last"
    
    # Build descriptive name for output files
    if args.exp is not None:
        # Legacy naming
        name_prefix = f"{args.model}_{args.task}_exp{args.exp}"
    else:
        # New naming
        divergence_tag = "div" if args.divergence else "nodiv"
        images_tag = "_img" if args.images else ""
        name_prefix = f"{args.model}_{args.task}_{args.dataset_size}_{args.training_epochs}_{args.save_freq}_{divergence_tag}{images_tag}"
    
    # Create argparse Namespace object to pass to run_trained_agent
    eval_args = argparse.Namespace()
    eval_args.agent = model_path
    eval_args.n_rollouts = args.n_rollouts
    eval_args.horizon = args.horizon
    eval_args.seed = args.seed
    eval_args.env = None
    eval_args.render = False
    eval_args.video_skip = 5
    eval_args.dataset_obs = False
    eval_args.images = args.images
    
    # Add video recording if requested
    if args.video:
        video_dir = "eval_videos"
        if not args.divergence:
            video_dir = os.path.join(video_dir, "no_divergence")
        else:
            video_dir = os.path.join(video_dir, "divergence")
        os.makedirs(video_dir, exist_ok=True)
        
        video_filename = f"{name_prefix}_{epoch_str}_seed{args.seed}.mp4"
        video_path = os.path.join(video_dir, video_filename)
        
        eval_args.video_path = video_path
        eval_args.camera_names = args.camera_names
        print(f"Will save video to: {video_path}")
    else:
        eval_args.video_path = None
        eval_args.camera_names = ["agentview"]
    
    # Add data recording if requested (this also saves stats JSON)
    logger = None
    if args.save_data:
        data_dir = "eval_data"
        if not args.divergence:
            data_dir = os.path.join(data_dir, "no_divergence")
        else:
            data_dir = os.path.join(data_dir, "divergence")
        os.makedirs(data_dir, exist_ok=True)
        
        data_filename = f"{name_prefix}_{epoch_str}_seed{args.seed}.hdf5"
        data_path = os.path.join(data_dir, data_filename)
        stats_path = data_path.replace(".hdf5", "_stats.json")
        log_path = data_path.replace(".hdf5", "_log.txt")
        
        eval_args.dataset_path = data_path
        eval_args.dataset_obs = True
        print(f"Will save data to: {data_path}")
        print(f"Will save stats to: {stats_path}")
        print(f"Will save logs to: {log_path}")
        
        # Redirect stdout to both console and log file
        logger = Logger(log_path)
        sys.stdout = logger
    else:
        eval_args.dataset_path = None
    
    # Fix checkpoint format issues if needed
    fix_checkpoint_camera_names(model_path)
    
    # Run evaluation directly
    print("\nRunning evaluation...")
    sys.stdout.flush()
    
    try:
        run_trained_agent(eval_args)
    finally:
        # Restore stdout and close log file
        if logger is not None:
            sys.stdout = logger.terminal
            logger.close()

def eval_model_loop(args):
    """Loop through multiple epochs and evaluate each checkpoint.
    
    Args:
        args: argparse.Namespace with evaluation arguments including loop parameters
    """
    # Determine evaluation frequency
    eval_freq = args.eval_freq if args.eval_freq is not None else args.save_freq
    
    # Validate eval_freq is a multiple of save_freq
    if eval_freq % args.save_freq != 0:
        print(f"Error: eval_freq ({eval_freq}) must be a multiple of save_freq ({args.save_freq})")
        sys.exit(1)
    
    # Determine epoch range
    start_epoch = args.start_epoch if args.start_epoch is not None else args.save_freq
    end_epoch = args.end_epoch if args.end_epoch is not None else args.training_epochs
    
    # Validate epoch range
    if start_epoch < args.save_freq:
        print(f"Error: start_epoch ({start_epoch}) must be >= save_freq ({args.save_freq})")
        sys.exit(1)
    
    if end_epoch > args.training_epochs:
        print(f"Error: end_epoch ({end_epoch}) must be <= training_epochs ({args.training_epochs})")
        sys.exit(1)
    
    if start_epoch > end_epoch:
        print(f"Error: start_epoch ({start_epoch}) must be <= end_epoch ({end_epoch})")
        sys.exit(1)
    
    # Generate list of epochs to evaluate
    epochs = list(range(start_epoch, end_epoch + 1, eval_freq))
    
    # Print configuration
    divergence_str = "with divergence" if args.divergence else "without divergence"
    images_str = "with images" if args.images else ""
    print("=" * 80)
    print(f"Running evaluation loop for {args.model} model {divergence_str} {images_str}")
    print(f"Task: {args.task}")
    print(f"Dataset size: {args.dataset_size}")
    print(f"Training epochs: {args.training_epochs}")
    print(f"Save frequency: {args.save_freq}")
    print(f"Eval frequency: {eval_freq}")
    print(f"Seed: {args.seed}")
    print(f"Video: {'Yes' if args.video else 'No'}")
    print(f"Save data: {'Yes' if args.save_data else 'No'}")
    print(f"Evaluating epochs: {start_epoch} to {end_epoch} (step={eval_freq})")
    print(f"Total evaluations: {len(epochs)}")
    print("=" * 80)
    print()
    
    # Loop through epochs
    failed_epochs = []
    successful_epochs = []
    
    for i, epoch in tqdm(enumerate(epochs, 1), total=len(epochs), desc="Evaluating epochs", unit="epoch"):
        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(epochs)}] Evaluating epoch {epoch}")
        print("=" * 80)
        
        # Create a copy of args with the current epoch
        epoch_args = argparse.Namespace(**vars(args))
        epoch_args.eval_epoch = epoch
        
        try:
            eval_single_model(epoch_args)
            successful_epochs.append(epoch)
            print(f"\n✓ Successfully completed evaluation for epoch {epoch}")
        except Exception as e:
            print(f"\n✗ Failed evaluation for epoch {epoch}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed_epochs.append(epoch)
        
        print()
    
    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION LOOP COMPLETE")
    print("=" * 80)
    print(f"Total evaluations: {len(epochs)}")
    print(f"Successful: {len(successful_epochs)}")
    print(f"Failed: {len(failed_epochs)}")
    
    if successful_epochs:
        print(f"\n✓ Successful epochs: {successful_epochs}")
    
    if failed_epochs:
        print(f"\n⚠ Failed epochs: {failed_epochs}")
        print(f"\nTo resume from first failed epoch, run:")
        loop_flag = "-LOOP" if args.loop else ""
        video_flag = "-V" if args.video else ""
        save_data_flag = "-SD" if args.save_data else ""
        cdm_flag = "-CDM" if args.divergence else ""
        img_flag = "-I" if args.images else ""
        eval_freq_flag = f"-EF {eval_freq}" if args.eval_freq is not None else ""
        print(f"python eval_rollouts.py {loop_flag} -M {args.model} {cdm_flag} {img_flag} "
              f"-T {args.task} -DS {args.dataset_size} "
              f"-TE {args.training_epochs} -SF {args.save_freq} {eval_freq_flag} -S {args.seed} "
              f"-START {min(failed_epochs)} {video_flag} {save_data_flag}")
    else:
        print("\n✓ All evaluations completed successfully!")
    
    print()


def main():
    """Main entry point for eval_rollouts.py script."""
    args = parse_args()
    
    if args.loop:
        # Loop mode: evaluate multiple epochs
        if args.eval_epoch is not None:
            print("Warning: -EE/--eval_epoch is ignored in loop mode")
        eval_model_loop(args)
    else:
        # Single evaluation mode
        if args.start_epoch is not None or args.end_epoch is not None or args.eval_freq is not None:
            print("Warning: -START/--start_epoch, -END/--end_epoch, and -EF/--eval_freq are only used in loop mode (-LOOP)")
        eval_single_model(args)


if __name__ == "__main__":
    main()

