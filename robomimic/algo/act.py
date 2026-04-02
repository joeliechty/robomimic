import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import robomimic.models.obs_nets as ObsNets
from robomimic.algo import register_algo_factory_func, PolicyAlgo
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

@register_algo_factory_func("act")
def algo_config_to_class(algo_config):
    """Factory function for the ACT algorithm"""
    return ACT, {}

class ACTNetwork(nn.Module):
    """
    Action Chunking Transformer (ACT) architecture.
    Uses a CVAE to encode multi-modal action trajectories, and a DETR-style
    Transformer (Encoder-Decoder) to predict action chunks.
    """
    def __init__(self, obs_shapes, ac_dim, chunk_size=100, hidden_dim=512, latent_dim=32, nheads=8, encoder_kwargs=None):
        super().__init__()
        self.obs_shapes = obs_shapes
        self.ac_dim = ac_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # 1. Standard robomimic observation encoder (handles images, proprioception, etc.)
        self.obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=OrderedDict(obs=OrderedDict(obs_shapes)),
            encoder_kwargs=encoder_kwargs,
        )
        obs_dim = self.obs_encoder.output_shape()[0]
        
        # Project observation features to transformer hidden dimension
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        
        # 2. CVAE Encoder (Compresses Action Chunk + Obs -> Latent z)
        self.cvae_encoder = nn.Sequential(
            nn.Linear(chunk_size * ac_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2) # Outputs mu and logvar
        )
        
        # 3. DETR-style PyTorch Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=4,
            num_decoder_layers=7,
            dim_feedforward=3200,
            dropout=0.1,
            batch_first=True
        )
        
        # Fixed positional embeddings for the decoder queries (the timeline of the action chunk)
        self.query_embed = nn.Embedding(chunk_size, hidden_dim)
        
        # 4. Action Head
        self.action_head = nn.Linear(hidden_dim, ac_dim)
        
    def forward_train(self, obs_dict, actions):
        """ Forward pass during training with CVAE encoding """
        B = actions.shape[0]
        
        # Encode observations
        obs_feat = self.obs_encoder(obs=obs_dict) # [B, obs_dim]
        obs_feat_proj = self.input_proj(obs_feat).unsqueeze(1) # [B, 1, hidden_dim]
        
        # CVAE Encoding
        flat_actions = actions.reshape(B, -1) # [B, chunk_size * ac_dim]
        cvae_input = torch.cat([flat_actions, obs_feat_proj.squeeze(1)], dim=-1)
        mu_logvar = self.cvae_encoder(cvae_input)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std # [B, latent_dim]
        
        # Prepare Transformer inputs
        # We pad z to hidden_dim to pass it as an encoder sequence item alongside obs
        z_proj = torch.zeros(B, 1, self.hidden_dim, device=z.device)
        z_proj[:, 0, :self.latent_dim] = z
        src = torch.cat([z_proj, obs_feat_proj], dim=1) # [B, Seq=2, hidden_dim]
        
        # Decoder queries are the fixed position embeddings
        tgt = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1) # [B, chunk_size, hidden_dim]
        
        # Run DETR Transformer
        hs = self.transformer(src, tgt) # [B, chunk_size, hidden_dim]
        
        # Predict Actions
        pred_actions = self.action_head(hs) # [B, chunk_size, ac_dim]
        
        return pred_actions, mu, logvar
        
    def forward_step(self, obs_dict):
        """ Forward pass during inference (samples z from prior) """
        B = list(obs_dict.values())[0].shape[0]
        
        # Sample z from standard normal prior
        obs_device = list(obs_dict.values())[0].device
        z = torch.randn(B, self.latent_dim, device=obs_device)
        
        obs_feat = self.obs_encoder(obs=obs_dict)
        obs_feat_proj = self.input_proj(obs_feat).unsqueeze(1)
        
        z_proj = torch.zeros(B, 1, self.hidden_dim, device=z.device)
        z_proj[:, 0, :self.latent_dim] = z
        src = torch.cat([z_proj, obs_feat_proj], dim=1)
        
        tgt = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        hs = self.transformer(src, tgt)
        
        return self.action_head(hs)

class ACT(PolicyAlgo):
    """
    Action Chunking Transformer Algorithm
    Handles loss computation (L1 + KL) and temporal ensembling.
    """
    def _create_networks(self):
        self.nets = nn.ModuleDict()
        self.chunk_size = self.algo_config.get("chunk_size", 100)
        self.nets["policy"] = ACTNetwork(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            chunk_size=self.chunk_size,
            hidden_dim=self.algo_config.get("hidden_dim", 512),
            latent_dim=self.algo_config.get("latent_dim", 32),
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        )
        self.nets = self.nets.float().to(self.device)
        self.action_history = []
        
    def process_batch_for_training(self, batch):
        """ Extracts current observation and chunk of future actions """
        input_batch = dict()
        # ACT conditions on a single observation (t=0)
        # Preserve all trailing dimensions (important for image observations).
        input_batch["obs"] = {k: batch["obs"][k][:, 0] for k in batch["obs"]}
        # Supervise on the whole chunk of actions
        input_batch["actions"] = batch["actions"][:, 0:self.chunk_size, :]
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        
    def _forward_training(self, batch):
        pred_actions, mu, logvar = self.nets["policy"].forward_train(batch["obs"], batch["actions"])
        return OrderedDict(actions=pred_actions, mu=mu, logvar=logvar)
        
    def _compute_losses(self, predictions, batch):
        losses = OrderedDict()
        pred_actions = predictions["actions"]
        true_actions = batch["actions"]
        
        # L1 Loss for Actions
        losses["l1_loss"] = F.l1_loss(pred_actions, true_actions)
        
        # KL Divergence for CVAE
        mu = predictions["mu"]
        logvar = predictions["logvar"]
        losses["kl_loss"] = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        kl_weight = self.algo_config.get("kl_weight", 10.0)
        losses["action_loss"] = losses["l1_loss"] + kl_weight * losses["kl_loss"]
        
        return losses
        
    def _train_step(self, losses):
        info = OrderedDict()
        info["policy_grad_norms"] = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"],
            max_grad_norm=self.global_config.train.max_grad_norm,
        )
        return info

    def log_info(self, info):
        log = super().log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["L1_Loss"] = info["losses"]["l1_loss"].item()
        log["KL_Loss"] = info["losses"]["kl_loss"].item()
        return log
        
    def reset(self):
        """ Clear prediction history on environment reset """
        self.action_history = []
        
    def get_action(self, obs_dict, goal_dict=None):
        """ 
        Query network every timestep and exponentially average overlapping predictions.
        """
        assert not self.nets.training
        
        # Predict the full chunk [chunk_size, ac_dim]
        pred_chunk = self.nets["policy"].forward_step(obs_dict)[0] 
        
        # Add to rolling history
        self.action_history.append(pred_chunk)
        if len(self.action_history) > self.chunk_size:
            self.action_history.pop(0)
            
        # Extract overlapping predictions for the *current* real-world timestep
        current_step_preds = []
        decay_k = self.algo_config.get("exp_decay_k", 0.01)
        weights = []
        
        for i, chunk in enumerate(reversed(self.action_history)):
            # 'chunk' is a prediction from 'i' steps ago.
            # Thus, the action for the *current* moment is at index 'i' in that chunk.
            current_step_preds.append(chunk[i])
            # Older predictions get exponentially lower weight
            weights.append(torch.exp(-decay_k * torch.tensor(i, dtype=torch.float32, device=self.device)))
            
        current_step_preds = torch.stack(current_step_preds)
        weights = torch.stack(weights).unsqueeze(1)
        weights = weights / weights.sum() # Normalize
        
        # Take the weighted average
        ensembled_action = (current_step_preds * weights).sum(dim=0)
        return ensembled_action