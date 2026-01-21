import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
import robomimic
import robosuite
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


class ClipEncoderWrapper(BaseNets.ConvBase):
    def __init__(self, input_shape=None, feature_dimension=512):
        super().__init__()
        self._feature_dim = feature_dimension

    def forward(self, obs):
        return obs.float()

    def output_shape(self, input_shape=None):
        return [self._feature_dim]


class IdentityPool(BaseNets.Module):
    def __init__(self, **kwargs): super(IdentityPool, self).__init__()
    def forward(self, x): return x
    def output_shape(self, input_shape): return input_shape


ObsUtils.OBS_ENCODER_BACKBONES["ClipEncoderWrapper"] = ClipEncoderWrapper
setattr(ObsCore, "ClipEncoderWrapper", ClipEncoderWrapper)
setattr(ObsCore, "IdentityPool", IdentityPool)


class CLIPRolloutWrapper(RolloutPolicy):
    """Wraps policy to add CLIP encoding at rollout time."""
    def __init__(self, policy, device, use_image_feats=False):
        print(">>> Entering CLIPRolloutWrapper.__init__()")
        #super().__init__(policy=policy)
        self.policy = policy
        self.device = device
        self.policy_device = device
        self.use_image_feats = use_image_feats
        self.clip_device = "cpu"  
        
        print(">>> use_image_feats =", self.use_image_feats)
        if self.use_image_feats:
            print(">>> About to import clip")
            import clip
            print(">>> clip imported")

            print(">>> About to load CLIP model")
            self.clip_model, _ = clip.load("ViT-B/32", device=self.clip_device)
            print(">>> CLIP model loaded")
            self.clip_model.eval()
            print(">>> CLIP set to eval")

            self.mean = torch.tensor(
                [0.48145466, 0.4578275, 0.40821073],
                device=self.clip_device
            ).view(1, 3, 1, 1)

            self.std = torch.tensor(
                [0.26862954, 0.26130258, 0.27577711],
                device=self.clip_device
            ).view(1, 3, 1, 1)
            print(">>> CLIP preprocessing tensors created")

    def set_eval(self):
        """Forward set_eval to the underlying policy."""
        if hasattr(self.policy, "set_eval"):
            self.policy.set_eval()

    def set_train(self):
        """Forward set_train to the underlying policy."""
        if hasattr(self.policy, "set_train"):
            self.policy.set_train()

    def start_episode(self):
        """Ensure the underlying policy starts its episode."""
        if hasattr(self.policy, "start_episode"):
            self.policy.start_episode()

    def __getattr__(self, name):
        """
        Failsafe: forward any attribute access (like 'device' or 'policy') 
        to the wrapped policy.
        """
        return getattr(self.policy, name)

    def __call__(self, ob, goal=None, batched_ob=False):
        """
        Process observation (add CLIP features if needed), then call parent policy.
        """
        if self.use_image_feats:
            img_key = "robot0_eye_in_hand_image"
            raw_img = None
            if hasattr(self.policy, "env"):
                # This bypasses the filtered 'ob' dict
                raw_obs = self.policy.env.get_observation()
                if img_key in raw_obs:
                    raw_img = raw_obs[img_key]
            img = torch.from_numpy(ob["robot0_eye_in_hand_image"]).float().to(self.clip_device)
            img = img.permute(2, 0, 1).unsqueeze(0).float() / 255.0
            print("Image shape before preprocessing:", img.shape)
            if img.ndim == 3:
                img = img.unsqueeze(0)

            
            img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
            
            img = (img - self.mean) / self.std

            
            with torch.no_grad():
                feats = self.clip_model.encode_image(img).float()

            
            ob["robot0_eye_in_hand_feats"] = feats.squeeze(0).cpu().numpy()
          
            del img, feats
            del ob[img_key]
        
        return self.policy.policy.get_action(obs_dict=ob, goal_dict=goal)



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
    
    
    base_path = "./exps/results/bc_rss/mlp"
    sub = "image_feats" if args.use_image_feats else "no_image_feats"
    config.train.output_dir = os.path.join(base_path, sub)
    config.train.output_dir += "_divergence" if args.use_divergence_loss else "_no_divergence"

    
    config.observation.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    if args.use_image_feats:
        
        config.observation.modalities.obs.low_dim.append("robot0_eye_in_hand_feats")
        
        config.observation.modalities.obs.rgb = []

    
    config.algo.loss.cdm_weight = args.div_loss_weight if args.use_divergence_loss else 0.0
    config.train.batch_size = 256  
    config.train.num_epochs = 200
    
   
    config.experiment.rollout.enabled = False
    config.experiment.save.enabled = True
    config.experiment.save.every_n_epochs = 3
    
   
    config.experiment.render_video = False
    config.experiment.rollout.n = 1
    
    if args.use_image_feats:
        config.experiment.env_meta_update_dict = {
            "use_images": True, 
            "use_camera_obs": True,
            "render_offscreen": True,
            "camera_names": ["robot0_eye_in_hand"], 
            "camera_height": 84, 
            "camera_width": 84
        }
    else:
        config.experiment.env_meta_update_dict = {"use_images": False}


ObsUtils.initialize_obs_utils_with_config(config)
model_keys = config.observation.modalities.obs.low_dim


original_run_rollout = TrainUtils.run_rollout
USE_IMAGE_FEATS = args.use_image_feats
def wrapped_run_rollout(*args, **kwargs):
    
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    
    env = kwargs.get("env", None)
    if env is not None and USE_IMAGE_FEATS:
        # This reaches into the robosuite/robomimic env and forces image generation
        if hasattr(env, "env"):
            # Set flags directly on the underlying robosuite env
            env.env.use_camera_obs = True
            env.env.camera_names = ["robot0_eye_in_hand"]
            env.env.camera_height = 84
            env.env.camera_width = 84

    base_policy = kwargs["policy"]
    
    if not USE_IMAGE_FEATS:
        return original_run_rollout(*args, **kwargs)

    
    if isinstance(base_policy, CLIPRolloutWrapper):
        return original_run_rollout(*args, **kwargs)

    
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