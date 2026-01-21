#!/usr/bin/env python
"""
Convenient script for evaluating trained models from the experiments.

Usage:
    python eval_model.py --model transformer --task lift --exp 0 --epoch 150
    python eval_model.py --model diffusion --task lift --exp 1 --epoch 100
    python eval_model.py --model transformer --task lift --exp 0 --epoch 150 --video
"""

import argparse
import os
import sys
from pathlib import Path
import torch

# Import the run_trained_agent function directly
from robomimic.scripts.run_trained_agent import run_trained_agent
import robomimic.algo.bc as bc
import robomimic.utils.tensor_utils as TensorUtils

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

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained robomimic models")
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        choices=["transformer", "mlp", "diffusion", "vae"],
        help="Model type: transformer (no divergence), transformer_cdm (with divergence), or diffusion"
    )
    
    parser.add_argument(
        "--task", "-t",
        type=str,
        required=True,
        choices=["lift", "can", "square"],
        help="Task name"
    )

    parser.add_argument(
        "--divergence", "-d",
        action="store_true",
        help="Use divergence model"
    )
    
    parser.add_argument(
        "--exp", "-e",
        type=int,
        required=True,
        help="Experiment number (e.g., 0 for exp0)"
    )
    
    parser.add_argument(
        "--epoch", "-p",
        type=int,
        default=None,
        help="Specific epoch to evaluate. If not provided, will use 'last.pth'"
    )
    
    parser.add_argument(
        "--n_rollouts", "-n",
        type=int,
        default=50,
        help="Number of evaluation rollouts (default: 50)"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=0,
        help="Random seed for evaluation (default: 0)"
    )
    
    parser.add_argument(
        "--video", "-v",
        action="store_true",
        help="Save evaluation video"
    )
    
    parser.add_argument(
        "--save_data",
        action="store_true",
        help="Save rollout data (.hdf5) and stats (.json)"
    )
    
    parser.add_argument(
        "--horizon",
        type=int,
        default=400,
        help="Maximum horizon for rollouts (default: 400)"
    )
    
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["agentview"],
        help="Camera names for video rendering"
    )
    
    return parser.parse_args()

def find_model_path(model_type, divergence, task, exp_num, epoch=None):
    """Find the path to the trained model checkpoint."""
    
    # Determine base directory based on model type
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
    
    # Construct path
    exp_dir = os.path.join(base_dir, task, f"exp{exp_num}")
    
    if not os.path.exists(exp_dir):
        print(f"Error: Experiment directory not found: {exp_dir}")
        sys.exit(1)
    
    # Find the timestamp subdirectory (there should be one)
    subdirs = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
    if len(subdirs) == 0:
        print(f"Error: No timestamp subdirectory found in {exp_dir}")
        sys.exit(1)
    
    timestamp_dir = os.path.join(exp_dir, subdirs[0])
    
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

def main():
    args = parse_args()
    
    # Find model checkpoint
    print(f"Looking for {args.model} model for task '{args.task}', experiment {args.exp}...")
    model_path = find_model_path(args.model, args.divergence, args.task, args.exp, args.epoch)
    print(f"Found model: {model_path}")
    
    # Build arguments for run_trained_agent
    epoch_str = f"epoch{args.epoch}" if args.epoch is not None else "last"
    
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
    
    # Add video recording if requested
    if args.video:
        video_dir = "eval_videos"
        if not args.divergence:
            video_dir = os.path.join(video_dir, "no_divergence")
        else:
            video_dir = os.path.join(video_dir, "divergence")
        os.makedirs(video_dir, exist_ok=True)
        
        video_filename = f"{args.model}_{args.task}_exp{args.exp}_{epoch_str}_seed{args.seed}.mp4"
        video_path = os.path.join(video_dir, video_filename)
        
        eval_args.video_path = video_path
        eval_args.camera_names = args.camera_names
        print(f"Will save video to: {video_path}")
    else:
        eval_args.video_path = None
        eval_args.camera_names = ["agentview"]
    
    # Add data recording if requested (this also saves stats JSON)
    if args.save_data:
        data_dir = "eval_data"
        if not args.divergence:
            data_dir = os.path.join(data_dir, "no_divergence")
        else:
            data_dir = os.path.join(data_dir, "divergence")
        os.makedirs(data_dir, exist_ok=True)
        
        data_filename = f"{args.model}_{args.task}_exp{args.exp}_{epoch_str}_seed{args.seed}.hdf5"
        data_path = os.path.join(data_dir, data_filename)
        stats_path = data_path.replace(".hdf5", "_stats.json")
        
        eval_args.dataset_path = data_path
        print(f"Will save data to: {data_path}")
        print(f"Will save stats to: {stats_path}")
    else:
        eval_args.dataset_path = None
    
    # Run evaluation directly
    print("\nRunning evaluation...")
    sys.stdout.flush()
    
    run_trained_agent(eval_args)

if __name__ == "__main__":
    main()
