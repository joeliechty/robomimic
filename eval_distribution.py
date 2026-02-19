#!/usr/bin/env python
"""
Script for evaluating action distribution predictions on training data.
This script loads a trained model and the training dataset, then generates
action predictions for the training observations to assess how well the model
captures the training action distribution.

Usage:
    python eval_distribution.py -M transformer -T lift -DS F -TE 500 -SF 5 -EE 500 -CDM
"""
import os
import sys
import argparse
import h5py
import torch
import numpy as np
from tqdm import tqdm

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.utils.file_utils import policy_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate action distribution predictions on training data")
    parser.add_argument("--model", "-M", type=str, required=True, 
                        choices=["transformer", "mlp", "diffusion", "diffusion_policy", "vae"], 
                        help="Model type")
    parser.add_argument("--task", "-T", type=str, required=True, 
                        choices=["lift", "can", "square"], 
                        help="Task name")
    parser.add_argument("--divergence", "-CDM", action="store_true", 
                        help="Use divergence model")
    parser.add_argument("--images", "-I", action="store_true", 
                        help="Model trained with images")
    parser.add_argument("--dataset_size", "-DS", type=str, required=True, 
                        choices=["F", "H1", "H2", "Q1", "Q2", "Q3", "Q4"], 
                        help="Dataset size: F (full), H1/H2 (half), Q1-Q4 (quarter)")
    parser.add_argument("--training_epochs", "-TE", type=int, required=True, 
                        help="Number of training epochs")
    parser.add_argument("--save_freq", "-SF", type=int, required=True, 
                        help="Model save frequency during training")
    parser.add_argument("--eval_epoch", "-EE", type=int, default=None, 
                        help="Specific epoch to evaluate. If not provided, will use 'last.pth'")
    parser.add_argument("--device", type=str, default="cuda:0", 
                        help="Device to use for inference (default: cuda:0)")
    # Loop mode arguments
    parser.add_argument("--loop", "-LOOP", action="store_true", 
                        help="Loop through all epochs from save_freq to training_epochs")
    parser.add_argument("--start_epoch", "-START", type=int, default=None, 
                        help="Starting epoch for loop mode (default: save_freq)")
    parser.add_argument("--end_epoch", "-END", type=int, default=None, 
                        help="Ending epoch for loop mode (default: training_epochs)")
    parser.add_argument("--eval_freq", "-EF", type=int, default=None, 
                        help="Evaluation frequency for loop mode (default: same as save_freq)")
    return parser.parse_args()


def find_model_path(model_type, divergence, images, task, dataset_size, training_epochs, save_freq, epoch=None):
    """Find the path to the trained model checkpoint."""
    
    # Determine base directory based on model type
    if model_type in ["diffusion", "diffusion_policy"]:
        base_dir = "robomimic/exps/results/diffusion_policy"
    elif model_type == "mlp":
        base_dir = "robomimic/exps/results/bc_rss/mlp_no_divergence"
    elif model_type == "transformer":
        if divergence:
            base_dir = "robomimic/exps/results/bc_rss/transformer_divergence"
        else:
            base_dir = "robomimic/exps/results/bc_rss/transformer_no_divergence"
    elif model_type == "vae":
        if divergence:
            base_dir = "robomimic/exps/results/bc_rss/vae_divergence"
        else:
            base_dir = "robomimic/exps/results/bc_rss/vae_no_divergence"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if images:
        base_dir += "_images"
    
    # Build experiment directory name
    exp_name = f"{dataset_size}_{training_epochs}_{save_freq}"
    exp_dir = os.path.join(base_dir, task, exp_name)
    
    if not os.path.exists(exp_dir):
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    
    # Find the timestamp subdirectory (use most recent if multiple exist)
    subdirs = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
    if len(subdirs) == 0:
        raise FileNotFoundError(f"No timestamp subdirectories found in {exp_dir}")
    
    timestamp_dir = os.path.join(exp_dir, sorted(subdirs)[-1])
    
    # Find model checkpoint
    if epoch is None:
        model_path = os.path.join(timestamp_dir, "last.pth")
    else:
        model_path = os.path.join(timestamp_dir, "models", f"model_epoch_{epoch}.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    return model_path


def find_training_dataset(task, dataset_size, divergence, images):
    """Find the path to the training dataset."""
    
    # Determine dataset filename
    if task == "lift":
        base_name = "low_dim_v15"
    elif task == "can":
        base_name = "can_feats"
    elif task == "square":
        base_name = "square_feats"
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Add dataset size suffix
    if dataset_size != "F":
        base_name = f"{base_name}_{dataset_size}"
    
    # Add divergence suffix if applicable
    if divergence:
        base_name = f"{base_name}_w_cdm"
    
    dataset_path = f"datasets/{task}/{base_name}.hdf5"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")
    
    return dataset_path


def get_output_directory(model_type, divergence, images, task, dataset_size, training_epochs, save_freq, epoch):
    """Get the output directory for saving predicted actions."""
    
    # Build model descriptor
    if model_type == "transformer":
        if divergence:
            model_desc = "transformer_divergence"
        else:
            model_desc = "transformer_no_divergence"
    elif model_type == "vae":
        if images:
            if divergence:
                model_desc = "vae_divergence_images"
            else:
                model_desc = "vae_no_divergence_images"
        else:
            if divergence:
                model_desc = "vae_divergence"
            else:
                model_desc = "vae_no_divergence"
    elif model_type in ["diffusion", "diffusion_policy"]:
        model_desc = "diffusion_policy"
    elif model_type == "mlp":
        model_desc = "mlp_no_divergence"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Build directory structure
    exp_name = f"{dataset_size}_{training_epochs}_{save_freq}"
    epoch_name = f"epoch_{epoch}" if epoch is not None else "last"
    
    output_dir = os.path.join("eval_data", "mmd_assessments", model_desc, task, exp_name, epoch_name)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


def predict_actions_for_dataset(model_path, dataset_path, device="cuda:0"):
    """
    Load model and dataset, then predict actions for all training observations.
    
    Args:
        model_path: Path to model checkpoint
        dataset_path: Path to training dataset
        device: Device to use for inference
        
    Returns:
        predicted_actions: Dictionary mapping demo keys to predicted action arrays
        dataset_meta: Metadata from the dataset
    """
    
    print(f"\nLoading model from: {model_path}")
    print(f"Loading dataset from: {dataset_path}")
    
    # Load the model
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=model_path,
        device=device,
        verbose=True
    )
    
    # Check if this is a transformer model that needs temporal context
    is_transformer = hasattr(policy.policy, 'algo_config') and \
                     hasattr(policy.policy.algo_config, 'transformer') and \
                     policy.policy.algo_config.transformer.enabled
    
    if is_transformer:
        context_length = policy.policy.algo_config.transformer.context_length
        print(f"\nTransformer model detected with context_length={context_length}")
    
    policy.start_episode()
    
    # Load the dataset
    f = h5py.File(dataset_path, 'r')
    demos = list(f["data"].keys())
    
    print(f"\nPredicting actions for {len(demos)} demonstrations...")
    
    predicted_actions = {}
    
    for demo_key in tqdm(demos, desc="Processing demos"):
        demo = f[f"data/{demo_key}"]
        
        # Get observations
        obs_keys = list(demo["obs"].keys())
        n_steps = demo["obs"][obs_keys[0]].shape[0]
        
        # Initialize observation history buffer for transformer models
        if is_transformer:
            obs_history = []
        
        # Predict actions for each timestep
        demo_actions = []
        
        for t in range(n_steps):
            # Build observation dictionary for this timestep
            obs_dict = {}
            for obs_key in obs_keys:
                obs_value = demo["obs"][obs_key][t]
                # Keep as numpy array - RolloutPolicy will handle conversion to tensor
                obs_dict[obs_key] = obs_value
            
            if is_transformer:
                # For transformer models, we need to provide temporal context
                # Add current observation to history
                obs_history.append(obs_dict)
                
                # Build input with proper temporal dimension [B, T, D]
                # Convert observations to tensors and add batch dimension
                obs_tensors = {}
                for obs_key in obs_keys:
                    # Stack history along time dimension
                    obs_sequence = [obs_history[i][obs_key] for i in range(len(obs_history))]
                    obs_sequence = np.stack(obs_sequence, axis=0)  # [T, D]
                    
                    # Pad if sequence is shorter than context_length
                    if len(obs_history) < context_length:
                        pad_len = context_length - len(obs_history)
                        # Pad with the first observation
                        first_obs = obs_sequence[0:1]  # [1, D]
                        padding = np.repeat(first_obs, pad_len, axis=0)  # [pad_len, D]
                        obs_sequence = np.concatenate([padding, obs_sequence], axis=0)  # [context_length, D]
                    
                    # Keep only the last context_length observations
                    if obs_sequence.shape[0] > context_length:
                        obs_sequence = obs_sequence[-context_length:]
                    
                    # Convert to tensor and add batch dimension: [B, T, D]
                    obs_tensor = torch.from_numpy(obs_sequence).float().unsqueeze(0).to(device)
                    obs_tensors[obs_key] = obs_tensor
                
                # Call the underlying policy directly with prepared tensors
                with torch.no_grad():
                    action_tensor = policy.policy.get_action(obs_dict=obs_tensors, goal_dict=None)
                    action = action_tensor.cpu().numpy()[0]  # Remove batch dimension
                
                # Maintain context window
                if len(obs_history) > context_length:
                    obs_history.pop(0)
            else:
                # For non-transformer models, use standard RolloutPolicy call
                action = policy(ob=obs_dict)
            
            # Action is numpy array
            demo_actions.append(action)
        
        # Stack actions for this demo
        predicted_actions[demo_key] = np.array(demo_actions)
        
        # Reset policy between demos
        policy.start_episode()
    
    # Extract dataset metadata
    dataset_meta = {
        "demos": demos,
        "total": len(demos)
    }
    
    # Copy env_args if available
    if "data" in f and len(demos) > 0:
        first_demo = f[f"data/{demos[0]}"]
        if "env_args" in first_demo.attrs:
            dataset_meta["env_args"] = dict(first_demo.attrs["env_args"])
    
    f.close()
    
    return predicted_actions, dataset_meta


def save_predicted_actions(predicted_actions, dataset_meta, output_path, source_dataset_path):
    """
    Save predicted actions to HDF5 file by copying the entire source dataset
    and replacing only the actions.
    
    Args:
        predicted_actions: Dictionary mapping demo keys to predicted action arrays
        dataset_meta: Metadata from the dataset
        output_path: Path to save the HDF5 file
        source_dataset_path: Path to the source training dataset to copy from
    """
    
    print(f"\nCreating dataset with predicted actions: {output_path}")
    print(f"Copying structure from: {source_dataset_path}")
    
    # Open source dataset for reading
    with h5py.File(source_dataset_path, 'r') as src_f:
        # Open destination file for writing
        with h5py.File(output_path, 'w') as dst_f:
            # Copy all top-level attributes
            for attr_key, attr_value in src_f.attrs.items():
                dst_f.attrs[attr_key] = attr_value
            
            # Copy the data group structure
            if "data" in src_f:
                data_grp = dst_f.create_group("data")
                
                # Copy each demonstration
                for demo_key in src_f["data"].keys():
                    src_demo = src_f["data"][demo_key]
                    dst_demo = data_grp.create_group(demo_key)
                    
                    # Copy all attributes
                    for attr_key, attr_value in src_demo.attrs.items():
                        dst_demo.attrs[attr_key] = attr_value
                    
                    # Copy observations group
                    if "obs" in src_demo:
                        obs_grp = dst_demo.create_group("obs")
                        for obs_key in src_demo["obs"].keys():
                            obs_data = src_demo["obs"][obs_key][()]
                            obs_grp.create_dataset(obs_key, data=obs_data, compression="gzip")
                    
                    # Copy next_obs group if it exists
                    if "next_obs" in src_demo:
                        next_obs_grp = dst_demo.create_group("next_obs")
                        for obs_key in src_demo["next_obs"].keys():
                            obs_data = src_demo["next_obs"][obs_key][()]
                            next_obs_grp.create_dataset(obs_key, data=obs_data, compression="gzip")
                    
                    # Replace actions with predicted actions
                    if demo_key in predicted_actions:
                        dst_demo.create_dataset("actions", data=predicted_actions[demo_key], compression="gzip")
                    else:
                        # Fall back to copying original actions if predictions are missing
                        print(f"Warning: No predictions found for {demo_key}, copying original actions")
                        if "actions" in src_demo:
                            dst_demo.create_dataset("actions", data=src_demo["actions"][()], compression="gzip")
                    
                    # Copy rewards if they exist
                    if "rewards" in src_demo:
                        dst_demo.create_dataset("rewards", data=src_demo["rewards"][()], compression="gzip")
                    
                    # Copy dones if they exist
                    if "dones" in src_demo:
                        dst_demo.create_dataset("dones", data=src_demo["dones"][()], compression="gzip")
                    
                    # Copy states if they exist
                    if "states" in src_demo:
                        dst_demo.create_dataset("states", data=src_demo["states"][()], compression="gzip")
            
            # Copy mask group if it exists
            if "mask" in src_f:
                src_f.copy("mask", dst_f)
    
    print(f"Saved {len(predicted_actions)} demonstrations with predicted actions")


def eval_single_epoch(args):
    """Evaluate action predictions for a single epoch."""
    
    print("="*80)
    print("EVALUATING ACTION DISTRIBUTION PREDICTIONS")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print(f"Dataset size: {args.dataset_size}")
    print(f"Training epochs: {args.training_epochs}")
    print(f"Save frequency: {args.save_freq}")
    print(f"Eval epoch: {args.eval_epoch if args.eval_epoch else 'last'}")
    print(f"Divergence: {args.divergence}")
    print(f"Images: {args.images}")
    print("="*80)
    
    # Find paths
    model_path = find_model_path(
        args.model, args.divergence, args.images, args.task, 
        args.dataset_size, args.training_epochs, args.save_freq, args.eval_epoch
    )
    
    dataset_path = find_training_dataset(
        args.task, args.dataset_size, args.divergence, args.images
    )
    
    output_dir = get_output_directory(
        args.model, args.divergence, args.images, args.task,
        args.dataset_size, args.training_epochs, args.save_freq, args.eval_epoch
    )
    
    output_path = os.path.join(output_dir, "predicted_actions.hdf5")
    
    # Check if already exists
    if os.path.exists(output_path):
        print(f"\nWARNING: Output file already exists: {output_path}")
        print("Overwriting...")
    
    # Predict actions
    predicted_actions, dataset_meta = predict_actions_for_dataset(
        model_path, dataset_path, device=args.device
    )
    
    # Save predictions (copying full dataset structure)
    save_predicted_actions(predicted_actions, dataset_meta, output_path, dataset_path)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    return output_path


def eval_loop(args):
    """Loop through multiple epochs and evaluate each."""
    
    # Determine evaluation frequency
    eval_freq = args.eval_freq if args.eval_freq is not None else args.save_freq
    
    # Validate eval_freq is a multiple of save_freq
    if eval_freq % args.save_freq != 0:
        raise ValueError(f"eval_freq ({eval_freq}) must be a multiple of save_freq ({args.save_freq})")
    
    # Determine epoch range
    start_epoch = args.start_epoch if args.start_epoch is not None else args.save_freq
    end_epoch = args.end_epoch if args.end_epoch is not None else args.training_epochs
    
    # Validate epoch range
    if start_epoch < args.save_freq:
        raise ValueError(f"start_epoch ({start_epoch}) must be >= save_freq ({args.save_freq})")
    
    if end_epoch > args.training_epochs:
        raise ValueError(f"end_epoch ({end_epoch}) must be <= training_epochs ({args.training_epochs})")
    
    if start_epoch > end_epoch:
        raise ValueError(f"start_epoch ({start_epoch}) must be <= end_epoch ({end_epoch})")
    
    # Generate list of epochs to evaluate
    epochs = list(range(start_epoch, end_epoch + 1, eval_freq))
    
    # Print configuration
    divergence_str = "with divergence" if args.divergence else "without divergence"
    images_str = "with images" if args.images else ""
    print("="*80)
    print(f"RUNNING EVALUATION LOOP")
    print("="*80)
    print(f"Model: {args.model} {divergence_str} {images_str}")
    print(f"Task: {args.task}")
    print(f"Dataset size: {args.dataset_size}")
    print(f"Training epochs: {args.training_epochs}")
    print(f"Save frequency: {args.save_freq}")
    print(f"Eval frequency: {eval_freq}")
    print(f"Evaluating epochs: {start_epoch} to {end_epoch} (step={eval_freq})")
    print(f"Total evaluations: {len(epochs)}")
    print("="*80)
    print()
    
    # Loop through epochs
    failed_epochs = []
    successful_epochs = []
    
    for i, epoch in enumerate(epochs, 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch} ({i}/{len(epochs)})")
        print(f"{'='*80}")
        
        try:
            # Create temporary args with specific epoch
            epoch_args = argparse.Namespace(**vars(args))
            epoch_args.eval_epoch = epoch
            epoch_args.loop = False
            
            output_path = eval_single_epoch(epoch_args)
            successful_epochs.append((epoch, output_path))
            print(f"✓ Epoch {epoch} completed successfully")
            
        except Exception as e:
            print(f"✗ Epoch {epoch} failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed_epochs.append((epoch, str(e)))
    
    # Summary
    print("\n" + "="*80)
    print("EVALUATION LOOP COMPLETE")
    print("="*80)
    print(f"Total evaluations: {len(epochs)}")
    print(f"Successful: {len(successful_epochs)}")
    print(f"Failed: {len(failed_epochs)}")
    
    if successful_epochs:
        print("\nSuccessful epochs:")
        for epoch, output_path in successful_epochs:
            print(f"  Epoch {epoch}: {output_path}")
    
    if failed_epochs:
        print("\nFailed epochs:")
        for epoch, error in failed_epochs:
            print(f"  Epoch {epoch}: {error}")
    
    print()


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.loop:
        eval_loop(args)
    else:
        eval_single_epoch(args)


if __name__ == "__main__":
    main()
