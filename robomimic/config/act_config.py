from robomimic.config.base_config import BaseConfig

class ACTConfig(BaseConfig):
    ALGO_NAME = "act"

    def train_config(self):
        """
        ACT, like BC, doesn't need "next_obs" from hdf5 - so save on storage and compute by disabling it.
        """
        super(ACTConfig, self).train_config()
        self.train.hdf5_load_next_obs = False

    def algo_config(self):
        """
        Populates the `config.algo` attribute of the config. Any parameter that ACT 
        needs to determine its training and test-time behavior should be populated here.
        """
        # Optimization parameters (ACT usually uses AdamW)
        self.algo.optim_params.policy.optimizer_type = "adamw"
        self.algo.optim_params.policy.learning_rate.initial = 1e-4      
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [] 
        self.algo.optim_params.policy.learning_rate.scheduler_type = "multistep" 
        self.algo.optim_params.policy.learning_rate.do_not_lock_keys()
        self.algo.optim_params.policy.regularization.L2 = 1e-4          

        # ACT specific architecture and training parameters
        self.algo.chunk_size = 100          # Number of actions to predict per forward pass
        self.algo.hidden_dim = 512          # Dimension of Transformer embeddings
        self.algo.latent_dim = 32           # Dimension of the CVAE latent space (z)
        self.algo.kl_weight = 10.0          # Weight of KL divergence loss in CVAE
        self.algo.exp_decay_k = 0.01        # Exponential decay rate for temporal ensembling