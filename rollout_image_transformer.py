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

# --- 1. THE TRANSFORMER EVAL WRAPPER ---
class TransformerEvalWrapper:
    def __init__(self, policy, device):
        self.policy = policy
        self.device = device
        
        # Determine model requirements
        self.expected_keys = list(policy.policy.obs_key_shapes.keys())
        self.context_length = policy.policy.algo_config.transformer.context_length
        self.needs_clip = "robot0_eye_in_hand_feats" in self.expected_keys
        
        # Buffer for temporal context: { key: [t1, t2, ... t10] }
        self.obs_history = {}

        if self.needs_clip:
            print(f"Loading CLIP for visual features...")
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            self.clip_model.eval()

    def _get_clip_feats(self, img_numpy):
        pil_img = Image.fromarray(img_numpy.astype('uint8'))
        img_tensor = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.clip_model.encode_image(img_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).cpu().numpy()

    def reset(self):
        """Clears the temporal history for a new episode."""
        self.obs_history = {}

    def __call__(self, ob):
        # 1. Generate CLIP features if needed
        if self.needs_clip:
            ob["robot0_eye_in_hand_feats"] = self._get_clip_feats(ob["robot0_eye_in_hand_image"])
        
        # 2. Update history buffer
        for k in self.expected_keys:
            if k not in self.obs_history:
                self.obs_history[k] = []
            self.obs_history[k].append(ob[k])
            
            # Maintain sliding window
            if len(self.obs_history[k]) > self.context_length:
                self.obs_history[k].pop(0)

        # 3. Prepare Batch [B=1, T=Context, D=Dim]
        input_obs = {}
        for k, v_list in self.obs_history.items():
            # Convert list to tensor and add batch dim
            t_seq = torch.from_numpy(np.array(v_list)).float().to(self.device).unsqueeze(0)
            
            # Pad if sequence is shorter than context_length (start of episode)
            current_len = t_seq.shape[1]
            if current_len < self.context_length:
                pad_len = self.context_length - current_len
                first_frame = t_seq[:, 0:1, :]
                padding = first_frame.repeat(1, pad_len, 1)
                t_seq = torch.cat([padding, t_seq], dim=1)
            
            input_obs[k] = t_seq

        # 4. Inference
        with torch.no_grad():
            # Use the underlying policy network directly to bypass rollout wrappers if necessary
            action = self.policy.policy.get_action(input_obs)
            
        # Return as numpy for environment
        return action.cpu().numpy()[0]

# --- 2. MAIN EVALUATION LOOP ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_rollouts", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=400)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=args.checkpoint, device=device)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    ObsUtils.initialize_obs_utils_with_config(config)

    # 2. Setup Environment
    env_meta = ckpt_dict["env_metadata"]
    env_meta["use_images"] = True
    env_meta["env_kwargs"]["use_camera_obs"] = True
    env_meta["env_kwargs"]["camera_names"] = ["agentview", "robot0_eye_in_hand"]
    env_meta["env_kwargs"]["camera_heights"] = [256, 84]
    env_meta["env_kwargs"]["camera_widths"] = [256, 84]
    
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=True
    )

    # 3. Wrap for Transformer context
    wrapped_policy = TransformerEvalWrapper(policy, device)

    # 4. Setup Video
    video_path = os.path.join(os.path.dirname(args.checkpoint), "eval_transformer.mp4")
    video_writer = imageio.get_writer(video_path, fps=20)

    successes = 0
    for ep in range(args.n_rollouts):
        print(f"Starting Episode {ep + 1}...")
        obs = env.reset()
        wrapped_policy.reset()
        
        for step in range(args.horizon):
            # Capture Video Frame (Agentview)
            raw_obs = env.env._get_observations()
            frame = np.flipud(raw_obs["agentview_image"])
            video_writer.append_data(frame)

            # Get Action
            action = wrapped_policy(obs)
            
            # Step
            obs, reward, done, info = env.step(action)
            
            if env.is_success()["task"]:
                print(f"Episode {ep+1}: SUCCESS")
                successes += 1
                break
            if done: break
        
    video_writer.close()
    print(f"Final Success Rate: {successes / args.n_rollouts:.2f}")

if __name__ == "__main__":
    main()