import os
import torch
import torch.nn.functional as F
import argparse
import numpy as np
import gc
import clip

from robomimic.config import config_factory
from robomimic.utils.train_utils import run_rollout
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.obs_utils import initialize_obs_utils_with_config
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils


# --- 1. THE ON-THE-FLY ENCODER WRAPPER ---
class CLIPOnTheFlyWrapper(RolloutPolicy):
    def __init__(self, policy, device):
        self.policy = policy
        self.device = device
        print(f"Loading CLIP ViT-B/32 on {device}...")
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()

        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    def _get_feats(self, img_numpy):
        # Convert HWC to NCHW and ensure it's on the correct device immediately
        img_tensor = torch.from_numpy(img_numpy.copy()).float().to(self.device).permute(2, 0, 1).unsqueeze(0)
        img_tensor /= 255.0 

        img_tensor = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        img_tensor = (img_tensor - self.mean) / self.std

        with torch.no_grad():
            feats = self.clip_model.encode_image(img_tensor)
        return feats.squeeze(0).cpu().numpy()

    def __call__(self, ob, goal=None, **kwargs):
        if "robot0_eye_in_hand_image" in ob:
            ob["robot0_eye_in_hand_feats"] = self._get_feats(ob["robot0_eye_in_hand_image"])
        return self.policy(ob=ob, goal=goal, **kwargs)

    def __getattr__(self, name):
        return getattr(self.policy, name)

# --- 2. MAIN EXECUTION ---

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--n_rollouts", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video_path", type=str, default="rollout.mp4")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fix for Docker/Headless environments
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    ckpt = torch.load(args.checkpoint, map_location=device)
    config = config_factory(algo_name="bc", image_feats=True)

    with config.values_unlocked():
        print("\n--- Available Experiment Keys ---")
        print(config.experiment.keys())
        config.train.data = args.dataset_path
        
        # --- ENABLE INTERNAL VIDEO ---
        config.experiment.render_video = True
        config.experiment.save.enabled = True
        
        config.experiment.env_meta_update_dict = {
            "use_images": True,
            "camera_names": ["robot0_eye_in_hand"],
            "camera_height": 84,
            "camera_width": 84,
        }
        config.observation.modalities.obs.low_dim = [
            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object", "robot0_eye_in_hand_feats"
        ]
        config.observation.modalities.obs.rgb = []

    initialize_obs_utils_with_config(config)

    env = EnvUtils.create_env_from_metadata(
        env_meta=ckpt["env_metadata"],
        render=args.render, 
        render_offscreen=True, 
        use_image_obs=True,     
    )
    dummy_obs = env.reset()
    obs_key_shapes = { k: v.shape for k, v in dummy_obs.items() }
    obs_key_shapes["robot0_eye_in_hand_feats"] = (512,)
    
    
    algo = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=env.observation_spec(),
        ac_dim=env.action_dimension,
        device=device,
    )
    algo.deserialize(ckpt["model"])
    
    policy = CLIPOnTheFlyWrapper(algo.policy, device)
    policy.set_eval()

    # When render_video is True, run_rollout will try to create its own writer
    # if video_writer is None. 
    rollout_stats = run_rollout(
        policy=policy,
        env=env,
        horizon=400,
        render=args.render,
        video_writer=None, # Internal system will now initialize this
        num_episodes=args.n_rollouts,
        device=device,
    )

    print("\nSuccess Rate:", rollout_stats["Success_Rate"])

if __name__ == "__main__":
    main()