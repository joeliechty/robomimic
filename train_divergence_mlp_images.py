import robomimic
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import json
import argparse
import numpy as np
import gc

import robomimic.models.base_nets as BaseNets
import robomimic.models.obs_core as ObsCore
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.config import config_factory
from robomimic.scripts.train import train
from robomimic.algo import RolloutPolicy

# --- 1. CLIP ENCODER & CUSTOM LAYERS ---
class ClipEncoderWrapper(BaseNets.ConvBase):
    def __init__(self, input_shape=None, feature_dimension=512):
        super().__init__()
        self._feature_dim = feature_dimension
        # No clip.load here
        # We'll just return the features as-is from dataset

    def forward(self, obs):
        # obs already contains precomputed features
        return obs.float()

    def output_shape(self, input_shape=None):
        return [self._feature_dim]


class IdentityPool(BaseNets.Module):
    def __init__(self, **kwargs): super(IdentityPool, self).__init__()
    def forward(self, x): return x
    def output_shape(self, input_shape): return input_shape

# Register with robomimic
ObsUtils.OBS_ENCODER_BACKBONES["ClipEncoderWrapper"] = ClipEncoderWrapper
setattr(ObsCore, "ClipEncoderWrapper", ClipEncoderWrapper)
setattr(ObsCore, "IdentityPool", IdentityPool)

# --- 2. UNIVERSAL ROLLOUT BRIDGE ---
class CLIPRolloutWrapper(RolloutPolicy):
    """Wraps policy to add CLIP encoding at rollout time."""
    def __init__(self, policy, device, use_image_feats=False):
        # Initialize parent class
        super().__init__(policy=policy)
        self.policy_device = device
        self.use_image_feats = use_image_feats
        self.clip_device = "cpu"  # CLIP runs on CPU to save GPU memory on M4
        
        if self.use_image_feats:
            import clip
            # Load CLIP on CPU for inference to avoid memory issues
            self.clip_model, _ = clip.load("ViT-B/32", device=self.clip_device)
            self.clip_model.eval()

            self.mean = torch.tensor(
                [0.48145466, 0.4578275, 0.40821073],
                device=self.clip_device
            ).view(1, 3, 1, 1)

            self.std = torch.tensor(
                [0.26862954, 0.26130258, 0.27577711],
                device=self.clip_device
            ).view(1, 3, 1, 1)

    def __call__(self, ob, goal=None, batched_ob=False):
        """
        Process observation (add CLIP features if needed), then call parent policy.
        """
        if self.use_image_feats and "robot0_eye_in_hand_image" in ob:
            # Extract image and process on CPU (where CLIP is)
            img = torch.from_numpy(ob["robot0_eye_in_hand_image"]).float().to(self.clip_device)
            if img.ndim == 3:
                img = img.unsqueeze(0)

            # Resize to CLIP input size
            img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
            # Normalize with ImageNet stats
            img = (img - self.mean) / self.std

            # Encode with CLIP
            with torch.no_grad():
                feats = self.clip_model.encode_image(img).float()

            # Add encoded features to observation
            ob["robot0_eye_in_hand_feats"] = feats.squeeze(0).cpu().numpy()
            
            # Clear CPU memory
            del img, feats
        
        # Call parent class __call__ which handles normalization and calls policy
        return super().__call__(ob=ob, goal=goal, batched_ob=batched_ob)


# --- 3. CONFIGURATION ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", "-D", type=str, required=True)
    parser.add_argument("--use_image_feats", "-IF", action='store_true', default=False)
    parser.add_argument("--use_divergence_loss", "-CDM", action='store_true')
    parser.add_argument("--div_loss_weight", "-L", type=float, default=0.01)
    return parser.parse_args()

args = parse_args()
config = config_factory(algo_name="bc", image_feats=args.use_image_feats)

with config.values_unlocked():
    config.train.data = args.dataset_path
    
    # Path Logic
    base_path = "./exps/results/bc_rss/mlp"
    sub = "image_feats" if args.use_image_feats else "no_image_feats"
    config.train.output_dir = os.path.join(base_path, sub)
    config.train.output_dir += "_divergence" if args.use_divergence_loss else "_no_divergence"

    # Modalities
    config.observation.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    if args.use_image_feats:
        # During training: use PRE-COMPUTED features from HDF5 (faster)
        config.observation.modalities.obs.low_dim.append("robot0_eye_in_hand_feats")
        # Do NOT configure RGB - we'll use CLIP only at rollout time
        config.observation.modalities.obs.rgb = []

    # Training & Rollout Logic
    config.algo.loss.cdm_weight = args.div_loss_weight if args.use_divergence_loss else 0.0
    config.train.batch_size = 256  
    config.train.num_epochs = 200
    
    # --- DISABLE ROLLOUT & VIDEO RENDERING FOR M4 ---
    config.experiment.rollout.enabled = False
    config.experiment.save.enabled = True
    config.experiment.save.every_n_epochs = 50
    
    
    # DISABLE VIDEO RENDERING - causes memory issues on M4
    config.experiment.render_video = False
    config.experiment.rollout.n = 1
    
    if args.use_image_feats:
        config.experiment.env_meta_update_dict = {
            "use_images": True, 
            "camera_names": ["robot0_eye_in_hand"], 
            "camera_height": 84, 
            "camera_width": 84
        }
    else:
        config.experiment.env_meta_update_dict = {"use_images": False}

# --- 4. RUN ---
ObsUtils.initialize_obs_utils_with_config(config)
model_keys = config.observation.modalities.obs.low_dim

# Patch Rollout
original_run_rollout = TrainUtils.run_rollout
USE_IMAGE_FEATS = args.use_image_feats
def wrapped_run_rollout(*args, **kwargs):
    # Force garbage collection before rollout to free memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    base_policy = kwargs["policy"]

    # ---- If NOT using image feats, do NOTHING ----
    if not USE_IMAGE_FEATS:
        return original_run_rollout(*args, **kwargs)

    # ---- Prevent double wrapping ----
    if isinstance(base_policy, CLIPRolloutWrapper):
        return original_run_rollout(*args, **kwargs)

    # ---- Resolve device safely ----
    if hasattr(base_policy, "device"):
        device = base_policy.device
    elif hasattr(base_policy, "policy") and hasattr(base_policy.policy, "device"):
        device = base_policy.policy.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs["policy"] = CLIPRolloutWrapper(
        policy=base_policy,
        device=device,
        use_image_feats=USE_IMAGE_FEATS,
    )
    return original_run_rollout(*args, **kwargs)

TrainUtils.run_rollout = wrapped_run_rollout

print(f"Training started. Rollouts every {config.experiment.save.every_n_epochs} epochs.")
train(config, device="cuda" if torch.cuda.is_available() else "cpu")