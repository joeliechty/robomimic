import os
# SET EGL BEFORE ANY OTHER IMPORTS
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import torch
import torch.nn.functional as F
import argparse
import numpy as np
import clip
import imageio
from PIL import Image

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

# --- 1. THE SMART MODALITY WRAPPER ---
class PolicyEvaluationWrapper:
    def __init__(self, policy, device):
        self.policy = policy
        self.device = device
        
        # Determine what the model actually wants
        self.expected_keys = list(policy.policy.obs_key_shapes.keys())
        self.needs_clip = "robot0_eye_in_hand_feats" in self.expected_keys
        
        if self.needs_clip:
            print(f"Detected Image-Feature model. Loading CLIP...")
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            self.clip_model.eval()
            self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
            self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
        else:
            print(f"Detected Low-Dim model. CLIP will not be used.")

    # def _get_feats(self, img_numpy):
    #     img_tensor = torch.from_numpy(img_numpy.copy()).float().to(self.device).permute(2, 0, 1).unsqueeze(0)
    #     img_tensor /= 255.0 
    #     img_tensor = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
    #     img_tensor = (img_tensor - self.mean) / self.std
    #     with torch.no_grad():
    #         feats = self.clip_model.encode_image(img_tensor)
    #     return feats.squeeze(0).cpu().numpy()
    def _get_feats(self, img_numpy):
        # 1. Convert numpy (H, W, 3) to PIL Image
        # This is exactly what Image.fromarray(img) did in your training script
        pil_img = Image.fromarray(img_numpy.astype('uint8'))

        # 2. Use the native CLIP preprocess pipeline
        # This handles Resize, CenterCrop, and Normalization correctly
        # The output is a (3, 224, 224) tensor
        img_tensor = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)

        # 3. Encode and Normalize
        with torch.no_grad():
            feats = self.clip_model.encode_image(img_tensor)
            
            # CRITICAL: Replicate the L2 normalization from your training script
            # embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            
        return feats.squeeze(0).cpu().numpy()
    def __call__(self, ob):
        # 1. If the model needs CLIP, generate it
        if self.needs_clip:
            # Note: We assume the image is present because we enabled it in main()
            ob["robot0_eye_in_hand_feats"] = self._get_feats(ob["robot0_eye_in_hand_image"])
        
        # 2. STRICT FILTERING
        # This is the key for Low-Dim models: We DELETE the images from the dict
        # before passing 'ob' to the policy.
        filtered_ob = {k: ob[k] for k in self.expected_keys if k in ob}
        
        return self.policy(filtered_ob)

    def start_episode(self):
        if hasattr(self.policy, "start_episode"):
            self.policy.start_episode()

# --- 2. MANUAL EVALUATION LOOP ---

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_rollouts", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=400)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=args.checkpoint, device=device, verbose=True)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    ObsUtils.initialize_obs_utils_with_config(config)

    # 2. Create Environment
    # We ALWAYS enable images here so we can save the video, 
    # even if the model is Low-Dim.
    env_meta = ckpt_dict["env_metadata"]
    env_meta["use_images"] = True
    env_meta["use_camera_obs"] = True
    env_meta["render_offscreen"] = True
    env_meta["env_kwargs"]["has_offscreen_renderer"] = True
    env_meta["env_kwargs"]["camera_names"] = ["agentview", "robot0_eye_in_hand"]
    env_meta["env_kwargs"]["camera_heights"] = [256, 84]
    env_meta["env_kwargs"]["camera_widths"] = [256, 84]
    
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=True, 
        use_image_obs=True,     
    )

    # 3. Wrap Policy
    wrapped_policy = PolicyEvaluationWrapper(policy, device)

    # 4. Setup Video
    video_path = os.path.join(os.path.dirname(args.checkpoint), "eval_rollout.mp4")
    video_writer = imageio.get_writer(video_path, fps=20)

    successes = 0
    for ep in range(args.n_rollouts):
        print(f"Starting Episode {ep + 1}...")
        obs = env.reset()
        wrapped_policy.start_episode()
        
        ep_success = False
        for step in range(args.horizon):
            # Capture frame for video from raw simulator
            raw_obs = env.env._get_observations()
            frame = np.flipud(raw_obs["agentview_image"])
            video_writer.append_data(frame)

            # Get action (Wrapper handles filtering out images for Low-Dim models)
            action = wrapped_policy(obs)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            if env.is_success()["task"]:
                ep_success = True
                break
            if done:
                break
        
        print(f"Episode {ep+1}: {'SUCCESS' if ep_success else 'FAILED'}")
        if ep_success: successes += 1

    video_writer.close()
    print(f"Success Rate: {successes / args.n_rollouts:.2f}")
    print(f"Video saved to: {video_path}")

if __name__ == "__main__":
    main()