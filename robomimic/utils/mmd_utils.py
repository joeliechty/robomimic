import torch
from robomimic.utils.divergence_utils import _add_phase, _bin_data, _load_training_data
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from tf_utils import _quat_to_rot_mat
import torch
import numpy as np

def gaussian_kernel(x, y, sigma=1.0):
    """
    Computes the Gaussian (RBF) kernel between x and y.
    
    Args:
        x: Tensor of shape (batch_size_x, features)
        y: Tensor of shape (batch_size_y, features)
        sigma: Bandwidth parameter
    
    Returns:
        Kernel matrix of shape (batch_size_x, batch_size_y)
    """
    # Calculate pairwise squared Euclidean distances
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x*y^T
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    
    # This creates a matrix of distances between every sample in x and every sample in y
    dist_sq = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    
    # Clamp to zero to handle numerical errors (avoid negative distances)
    dist_sq = torch.clamp(dist_sq, min=0)
    
    # Compute the Gaussian kernel value
    return torch.exp(-dist_sq / (2 * sigma**2))

def mmd_loss(source_samples, target_samples, w_sigma=1.0):
    """
    Computes Maximum Mean Discrepancy (MMD) between two sets of samples
    using the Median Heuristic for bandwidth.
    """

    # put on cuda if available
    orig_device = source_samples.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_samples = source_samples.to(device)
    target_samples = target_samples.to(device)

    # 1. Concatenate all data to find a common scale for the heuristic
    # This ensures we measure "distance" consistently across both sets
    all_samples = torch.cat([source_samples, target_samples], dim=0)
    
    # 2. Compute pairwise distances for the heuristic
    # (We only need a subset if data is huge, but here we use all for accuracy)
    x_norm = (all_samples**2).sum(1).view(-1, 1)
    dist_sq = x_norm + x_norm.t() - 2.0 * torch.mm(all_samples, all_samples.t())
    dist_sq = torch.clamp(dist_sq, min=0)
    
    # 3. Median Heuristic: Set sigma to the median distance
    # This centers the kernel "sensitivity" on the typical spacing of your data.
    # We use sqrt because the kernel expects sigma, but we have distance squared.
    sigma = torch.median(torch.sqrt(dist_sq))
    
    # Fallback for identical data (distance 0) to avoid division by zero
    if sigma.item() == 0:
        sigma = torch.tensor(1.0)

    # weight the bandwidth if desired (e.g., w_sigma=1.0 means no change, w_sigma=0.5 makes it more sensitive)
    sigma *= w_sigma

    # 4. Compute Kernels
    # K_xx: Similarity among source samples
    # K_yy: Similarity among target samples
    # K_xy: Similarity between source and target
    k_xx = gaussian_kernel(source_samples, source_samples, sigma)
    k_yy = gaussian_kernel(target_samples, target_samples, sigma)
    k_xy = gaussian_kernel(source_samples, target_samples, sigma)
    
    # 5. Compute MMD^2
    # The formula is: E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    mmd_sq = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return mmd_sq.to(orig_device)
    
def bin_actions_as_twists(data, demo_keys, n_bins, osc_dt=0.02):
    """
    Extract actions and scale them using OSC dt to make them twists. Actions are binned with the action
    repeated until the next action is available. For example, if a traj has 5 actions but there are 10
    timesteps, each action will be repeated for 2 timesteps.
    Args:
        data: dataset dictionary
        demo_keys: list of demo keys in the order matching the binned tensors (from _bin_data)
        n_bins: number of bins to use for binning actions
        osc_dt: time delta to scale actions (default 0.02s for 50Hz OSC)
    Returns:
        action_tensor: Binned action tensor [n_demos, n_bins, action_dim]
    """
    n_demos = len(demo_keys)
    
    # Check action dimension from first demo
    first_actions = data['get_actions'](demo_keys[0])
    # Use first 6 dimensions if available (velocities + angular velocities), otherwise use all
    action_dim = min(6, first_actions.shape[1])
    
    # Initialize output tensor
    action_tensor = torch.full((n_demos, n_bins, action_dim), float('nan'))
    
    for i, demo_key in enumerate(demo_keys):
        # Get raw actions
        actions = data['get_actions'](demo_key)
        
        # Convert to torch tensor if numpy
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
            
        # Select dimensions and scale to twists
        # Actions are assumed to be deltas, so dividing by dt gives velocity/twist
        twists = actions[:, :action_dim].float() / osc_dt
        
        # Get phase for this demo
        phase = data[demo_key]['phase']
        
        # Calculate bin indices: phase [0, 1] -> [0, n_bins-1]
        bin_indices = torch.round(phase * (n_bins - 1)).long()
        
        # Ensure indices are within bounds
        bin_indices = torch.clamp(bin_indices, 0, n_bins - 1)
        
        # Place twists into bins
        # If multiple phases map to same bin, this takes the last one (consistent with overwriting)
        action_tensor[i, bin_indices] = twists

    # Fill in gaps (NaNs) with forward filling (zero-order hold)
    for i in range(n_demos):
        # Determine valid mask
        mask = ~torch.isnan(action_tensor[i]).any(dim=1)
        
        # Indices where we have data
        valid_indices = torch.where(mask)[0]
        
        if len(valid_indices) == 0:
            continue
            
        # 1. Fill beginning: if first valid index > 0, replicate first value backwards
        if valid_indices[0] > 0:
            action_tensor[i, :valid_indices[0]] = action_tensor[i, valid_indices[0]]
            
        # 2. Forward fill holes
        last_val = action_tensor[i, valid_indices[0]]
        for t in range(valid_indices[0] + 1, n_bins):
            if mask[t]:
                last_val = action_tensor[i, t]
            else:
                action_tensor[i, t] = last_val
                
    return action_tensor

def compute_rollout_mmd(args):
    # Load the dataset
    print("\n0.1. Loading training dataset...")
    training_data = _load_training_data(args.training_dataset)

    print("\n0.2. Loading rollout dataset...")
    rollout_data = _load_training_data(args.rollout_dataset)

    # Extract directory from dataset path for saving figures
    dataset_dir = os.path.dirname(args.rollout_dataset)
    print(f"\nFigures will be saved to: {dataset_dir}")

    print(f"\n0.3.1 Adding phases...")
    training_data = _add_phase(training_data)
    rollout_data = _add_phase(rollout_data)

    # calculate trajectory length stats
    n_actions = []
    for demo_key in training_data['demos']:
        actions = training_data['get_actions'](demo_key)
        n_actions.append(len(actions))
    train_mean_n_actions = np.mean(n_actions)
    train_std_n_actions = np.std(n_actions)
    train_min_n_actions = np.min(n_actions)
    train_max_n_actions = np.max(n_actions)
    train_median_n_actions = np.median(n_actions)
    
    print(f"\n0.8. Training trajectory length statistics across all demos:")
    print(f"   Mean: {train_mean_n_actions:.1f} timesteps")
    print(f"   Std:  {train_std_n_actions:.1f} timesteps")
    print(f"   Min:  {train_min_n_actions} timesteps")
    print(f"   Max:  {train_max_n_actions} timesteps")
    print(f"   Median: {train_median_n_actions} timesteps")

    n_actions = []
    for demo_key in rollout_data['demos']:
        actions = rollout_data['get_actions'](demo_key)
        n_actions.append(len(actions))
    rollout_mean_n_actions = np.mean(n_actions)
    rollout_std_n_actions = np.std(n_actions)
    rollout_min_n_actions = np.min(n_actions)
    rollout_max_n_actions = np.max(n_actions)
    rollout_median_n_actions = np.median(n_actions)
    
    print(f"\n0.8. Rollout trajectory length statistics across all demos:")
    print(f"   Mean: {rollout_mean_n_actions:.1f} timesteps")
    print(f"   Std:  {rollout_std_n_actions:.1f} timesteps")
    print(f"   Min:  {rollout_min_n_actions} timesteps")
    print(f"   Max:  {rollout_max_n_actions} timesteps")
    print(f"   Median: {rollout_median_n_actions} timesteps")

    max_n_actions = max(train_max_n_actions, rollout_max_n_actions)
    print(f"\n0.9. Using max_n_actions={max_n_actions} for binning actions for MMD computation.")

    # explore what keys are in the rollout data
    # print(f"\n0.10. Exploring rollout dataset keys and structure:")
    # for key in rollout_data.keys():
    #     print(f"   Key: {key}")
    #     if isinstance(rollout_data[key], dict):
    #         print(f"      Sub-keys: {list(rollout_data[key].keys())}")

    print(f"\n0.11. Binning data for training and rollout datasets...")
    print(f"   Training data...")
    t_phase_tensor, t_ee_state_tensor, t_demo_tensor_keys, t_nan_mask, t_dphase = _bin_data(training_data, n_bins=max_n_actions)
    print(f"   Rollout data...")
    r_phase_tensor, r_ee_state_tensor, r_demo_tensor_keys, r_nan_mask, r_dphase = _bin_data(rollout_data, n_bins=max_n_actions)
    print(f"   Done binning phase and ee_state data.")

    # extract actions and scale them using OSC dt to make them twists, then bin them
    print(f"\n0.12. Binning actions as twists for MMD computation...")
    print(f"   Training data...")
    t_action_tensor = bin_actions_as_twists(training_data, t_demo_tensor_keys, max_n_actions, osc_dt=args.osc_dt)   # [n_demos, n_bins, action_dim]
    print(f"   Rollout data...")
    r_action_tensor = bin_actions_as_twists(rollout_data, r_demo_tensor_keys, max_n_actions, osc_dt=args.osc_dt)    # [n_demos, n_bins, action_dim]
    print(f"   Done binning actions as twists.")

    # compute mmd between training and rollout actions
    print(f"\n1. Computing MMD between training and rollout actions...")
    # Compute MMD for each time step - store as CPU values immediately
    mmd_per_timestep = []
    for step in range(max_n_actions):
        mmd_step = mmd_loss(t_action_tensor[:, step, :], r_action_tensor[:, step, :], args.w_sigma)
        mmd_per_timestep.append(mmd_step.cpu().item())
        del mmd_step  # Delete immediately
                
    # Work with CPU numpy array instead of tensors
    mmd_per_timestep_np = np.array(mmd_per_timestep) # [max_n_actions]

    # Save MMD values to disk as csv
    mmd_csv_path = os.path.join(dataset_dir, "mmd_per_timestep.csv")
    np.savetxt(mmd_csv_path, mmd_per_timestep_np, delimiter=",")
    print(f"\nMMD per timestep saved to: {mmd_csv_path}")

    return mmd_per_timestep_np, t_action_tensor, r_action_tensor, t_phase_tensor, \
        r_phase_tensor, t_demo_tensor_keys, r_demo_tensor_keys, t_ee_state_tensor, \
        r_ee_state_tensor, t_nan_mask, r_nan_mask, t_dphase, r_dphase

def visualize_mmd_for_trajectories(
    training_ee_state_tensor,
    rollout_ee_state_tensor,
    training_actions_tensor,
    rollout_actions_tensor,
    mmd_tensor,
    rollout_demo_indices,
    demo_keys,
    nan_mask=None,
    phase_tensor=None,
    frame_skip=10,
    demo_indices=None,
    figsize=(24, 14),
    title="Trajectory Score Visualization"
):
    """
    Visualize several trajectories from the rollouts overlaid on all the training trajectories.
    Shows the selected rollouts as solid black lines on top of all others to understand
    how it sits within the data distribution. Also visualizes the mmd in a subplot running along
    the bottom. breaks down the actions into their components (linear xyz and angular xyz) to see
    which components with the rollout actions in plack lines overlayed on the training data actions.
    for the actions we have x,y,z linear and sine and cosine of roll, pitch, yaw to avoid discontinuities.
    
    Args:
        training_ee_state_tensor: torch.tensor [n_demos, n_bins, 7], end-effector poses (pos + quat)
        rollout_ee_state_tensor: torch.tensor [n_demos, n_bins, 7], end-effector poses (pos + quat)
        training_actions_tensor: torch.tensor [n_demos, n_bins, action_dim], binned and scaled actions for training data
        rollout_actions_tensor: torch.tensor [n_demos, n_bins, action_dim], binned and scaled actions for rollout data
        mmd_tensor: torch.tensor [n_bins], MMD values per time step
        rollout_demo_indices: list of rollout indices to visualize in black
        demo_keys: list of demo keys corresponding to each trajectory
        nan_mask: torch.tensor [n_demos, n_bins], boolean mask for NaN values (optional)
        phase_tensor: torch.tensor [n_demos, n_bins, 1], phase values (optional)
        frame_skip: int, plot coordinate frames every N steps (default 10)
        demo_indices: list of indices to visualize (if None, visualizes all demos)
        figsize: tuple, figure size (default (24, 14))
        title: str, overall figure title
    
    Returns:
        fig: matplotlib figure object
    """
    
    class Arrow3D(FancyArrowPatch):
        def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._xyz = (x, y, z)
            self._dxdydz = (dx, dy, dz)

        def do_3d_projection(self, renderer=None):
            x1, y1, z1 = self._xyz
            dx, dy, dz = self._dxdydz
            x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

            xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            
            return np.min(zs)
    
    # Select demos to visualize
    n_training_demos = training_ee_state_tensor.shape[0]
    n_rollout_demos = rollout_ee_state_tensor.shape[0]
    n_bins = training_ee_state_tensor.shape[1]
    
    if demo_indices is None:
        demo_indices = list(range(n_training_demos))
    
    # Convert to numpy for plotting
    training_ee_state_np = training_ee_state_tensor.cpu().numpy()
    rollout_ee_state_np = rollout_ee_state_tensor.cpu().numpy()
    training_actions_np = training_actions_tensor.cpu().numpy()  # [n_demos, n_bins, action_dim]
    rollout_actions_np = rollout_actions_tensor.cpu().numpy()  # [n_demos, n_bins, action_dim]
    mmd_np = mmd_tensor if isinstance(mmd_tensor, np.ndarray) else mmd_tensor.cpu().numpy()  # [n_bins]
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 7, hspace=0.4, wspace=0.4, height_ratios=[1, 1, 1, 0.6])
    
    # 3D trajectory plot (spans left 3 columns, all 3 top rows)
    ax_3d = fig.add_subplot(gs[:3, :3], projection='3d')
    
    # Action linear components (columns 3-4, 2 wide)
    ax_action_lin_x = fig.add_subplot(gs[0, 3:5])
    ax_action_lin_y = fig.add_subplot(gs[1, 3:5])
    ax_action_lin_z = fig.add_subplot(gs[2, 3:5])
    
    # Action angular components (columns 5-6, 2 wide)
    ax_action_ang_x = fig.add_subplot(gs[0, 5:7])
    ax_action_ang_y = fig.add_subplot(gs[1, 5:7])
    ax_action_ang_z = fig.add_subplot(gs[2, 5:7])
    
    # MMD subplot (4th row: spans all columns)
    ax_mmd = fig.add_subplot(gs[3, :])
    
    # Color map for different demos
    colors = plt.cm.tab20(np.linspace(0, 1, len(demo_indices)))
    
    # All demos shown with low transparency for context
    line_alpha = 0.15 if len(demo_indices) > 20 else 0.3
    
    # First pass: Plot all training trajectories and actions
    for idx, demo_idx in enumerate(demo_indices):
        demo_key = demo_keys[demo_idx]
        color = colors[idx]
        
        # Get trajectory for this demo
        traj = training_ee_state_np[demo_idx]  # [n_bins, 7]
        actions = training_actions_np[demo_idx]  # [n_bins, action_dim]
        
        # Get valid indices (non-NaN)
        if nan_mask is not None:
            valid_mask = ~nan_mask[demo_idx].cpu().numpy()
        else:
            valid_mask = ~np.isnan(traj).any(axis=1)
        
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            continue
        
        # Extract position and quaternion
        positions = traj[valid_indices, :3]  # [n_valid, 3]
        
        # Plot 3D trajectory with low alpha
        ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                   color=color, alpha=line_alpha, linewidth=1.0)
        
        # Extract phase values for plotting
        if phase_tensor is not None:
            phase_values = phase_tensor[demo_idx, valid_indices, 0].cpu().numpy()
        else:
            phase_values = valid_indices  # Fall back to time steps if no phase
        
        # Get actions for valid indices
        valid_actions = actions[valid_indices]  # [n_valid, action_dim]
        
        # Plot action linear components (first 3 dimensions)
        ax_action_lin_x.plot(phase_values, valid_actions[:, 0], color=color, alpha=line_alpha, linewidth=1.0)
        ax_action_lin_y.plot(phase_values, valid_actions[:, 1], color=color, alpha=line_alpha, linewidth=1.0)
        ax_action_lin_z.plot(phase_values, valid_actions[:, 2], color=color, alpha=line_alpha, linewidth=1.0)
        
        # Plot action angular components (next 3 dimensions)
        if valid_actions.shape[1] >= 6:
            ax_action_ang_x.plot(phase_values, valid_actions[:, 3], color=color, alpha=line_alpha, linewidth=1.0)
            ax_action_ang_y.plot(phase_values, valid_actions[:, 4], color=color, alpha=line_alpha, linewidth=1.0)
            ax_action_ang_z.plot(phase_values, valid_actions[:, 5], color=color, alpha=line_alpha, linewidth=1.0)
    
    # Second pass: Plot the selected rollout trajectories on top with dark colored lines
    # Create dark color palette for rollouts
    rollout_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for idx, selected_demo_idx in enumerate(rollout_demo_indices):
        rollout_color = rollout_colors[idx % len(rollout_colors)]
        selected_traj = rollout_ee_state_np[selected_demo_idx]  # [n_bins, 7]
        selected_actions = rollout_actions_np[selected_demo_idx]  # [n_bins, action_dim]
        selected_demo_key = demo_keys[selected_demo_idx] if selected_demo_idx < len(demo_keys) else f"rollout_{selected_demo_idx}"
        
        # Get valid indices for selected demo
        selected_valid_mask = ~np.isnan(selected_traj).any(axis=1)
        selected_valid_indices = np.where(selected_valid_mask)[0]
        
        if len(selected_valid_indices) == 0:
            continue
            
        # Extract position and quaternion for selected demo
        selected_positions = selected_traj[selected_valid_indices, :3]  # [n_valid, 3]
        selected_quaternions = selected_traj[selected_valid_indices, 3:]  # [n_valid, 4]
        
        # Plot selected trajectory in 3D with unique dark color
        ax_3d.plot(selected_positions[:, 0], selected_positions[:, 1], selected_positions[:, 2], 
                   color=rollout_color, alpha=1.0, linewidth=2.5, 
                   label=f"Rollout {selected_demo_idx}", zorder=100)
        
        # Plot coordinate frames for each rollout trajectory with smaller arrows
        frame_indices = selected_valid_indices[::frame_skip]
        for frame_idx in frame_indices:
            pos = selected_traj[frame_idx, :3]
            quat = torch.from_numpy(selected_traj[frame_idx, 3:]).unsqueeze(0)
            
            # Convert quaternion to rotation matrix
            rot_mat = _quat_to_rot_mat(quat, w_first=False).squeeze(0).numpy()
        
            # Draw coordinate frame axes (smaller to reduce clutter)
            axis_length = 0.015  # Reduced from 0.025 for less clutter
            
            # Extract each axis from rotation matrix columns
            x_axis = rot_mat[:, 0]
            y_axis = rot_mat[:, 1]
            z_axis = rot_mat[:, 2]
            
            # Normalize to ensure unit length
            x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
            y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
            z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
            
            # X-axis (red)
            ax_3d.add_artist(Arrow3D(
                pos[0], pos[1], pos[2],
                x_axis[0] * axis_length,
                x_axis[1] * axis_length,
                x_axis[2] * axis_length,
                mutation_scale=10, lw=1.5, arrowstyle='-|>', color='red', alpha=0.7
            ))
            
            # Y-axis (green)
            ax_3d.add_artist(Arrow3D(
                pos[0], pos[1], pos[2],
                y_axis[0] * axis_length,
                y_axis[1] * axis_length,
                y_axis[2] * axis_length,
                mutation_scale=10, lw=1.5, arrowstyle='-|>', color='green', alpha=0.7
            ))
            
            # Z-axis (blue)
            ax_3d.add_artist(Arrow3D(
                pos[0], pos[1], pos[2],
                z_axis[0] * axis_length,
                z_axis[1] * axis_length,
                z_axis[2] * axis_length,
                mutation_scale=10, lw=1.5, arrowstyle='-|>', color='blue', alpha=0.7
            ))
        
        # Extract phase values for selected trajectory
        if phase_tensor is not None:
            selected_phase_values = phase_tensor[selected_demo_idx, selected_valid_indices, 0].cpu().numpy()
        else:
            selected_phase_values = selected_valid_indices
        
        # Get actions for valid indices
        selected_valid_actions = selected_actions[selected_valid_indices]  # [n_valid, action_dim]
        
        # Plot selected trajectory action linear components with unique dark color
        ax_action_lin_x.plot(selected_phase_values, selected_valid_actions[:, 0], 
                            color=rollout_color, alpha=1.0, linewidth=2.5, zorder=100)
        ax_action_lin_y.plot(selected_phase_values, selected_valid_actions[:, 1], 
                            color=rollout_color, alpha=1.0, linewidth=2.5, zorder=100)
        ax_action_lin_z.plot(selected_phase_values, selected_valid_actions[:, 2], 
                            color=rollout_color, alpha=1.0, linewidth=2.5, zorder=100)
        
        # Plot selected trajectory action angular components with unique dark color
        if selected_valid_actions.shape[1] >= 6:
            ax_action_ang_x.plot(selected_phase_values, selected_valid_actions[:, 3], 
                                color=rollout_color, alpha=1.0, linewidth=2.5, zorder=100)
            ax_action_ang_y.plot(selected_phase_values, selected_valid_actions[:, 4], 
                                color=rollout_color, alpha=1.0, linewidth=2.5, zorder=100)
            ax_action_ang_z.plot(selected_phase_values, selected_valid_actions[:, 5], 
                                color=rollout_color, alpha=1.0, linewidth=2.5, zorder=100)
    
    # Plot MMD values
    mmd_phase_values = np.arange(len(mmd_np))
    if phase_tensor is not None:
        # Use phase values from 0 to 1
        mmd_phase_values = np.linspace(0, 1, len(mmd_np))
    
    ax_mmd.plot(mmd_phase_values, mmd_np, color='purple', linewidth=2.5, label='MMD')
    ax_mmd.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax_mmd.grid(True, alpha=0.3)
    ax_mmd.set_xlabel('Phase' if phase_tensor is not None else 'Time Step', fontsize=10)
    ax_mmd.set_ylabel('MMD', fontsize=10)
    ax_mmd.set_title('Maximum Mean Discrepancy (Training vs Rollout)', fontsize=11, fontweight='bold')
    ax_mmd.legend(loc='upper right', fontsize=9)
    
    # Configure 3D plot
    ax_3d.set_xlabel('X Position (m)')
    ax_3d.set_ylabel('Y Position (m)')
    ax_3d.set_zlabel('Z Position (m)')
    ax_3d.set_title('3D Trajectories (Selected in Black)', fontsize=12, fontweight='bold')
    ax_3d.legend(loc='upper right', fontsize=8)
    ax_3d.grid(True, alpha=0.3)
    
    # Configure action linear component subplots
    ax_action_lin_x.set_ylabel('Linear X (m/s)')
    ax_action_lin_x.set_title('Action: Linear X', fontsize=10, fontweight='bold')
    ax_action_lin_x.grid(True, alpha=0.3)
    ax_action_lin_x.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    
    ax_action_lin_y.set_ylabel('Linear Y (m/s)')
    ax_action_lin_y.set_title('Action: Linear Y', fontsize=10, fontweight='bold')
    ax_action_lin_y.grid(True, alpha=0.3)
    ax_action_lin_y.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    
    ax_action_lin_z.set_xlabel('Phase' if phase_tensor is not None else 'Time Step')
    ax_action_lin_z.set_ylabel('Linear Z (m/s)')
    ax_action_lin_z.set_title('Action: Linear Z', fontsize=10, fontweight='bold')
    ax_action_lin_z.grid(True, alpha=0.3)
    ax_action_lin_z.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    
    # Configure action angular component subplots
    ax_action_ang_x.set_ylabel('Angular X (rad/s)')
    ax_action_ang_x.set_title('Action: Angular X', fontsize=10, fontweight='bold')
    ax_action_ang_x.grid(True, alpha=0.3)
    ax_action_ang_x.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    
    ax_action_ang_y.set_ylabel('Angular Y (rad/s)')
    ax_action_ang_y.set_title('Action: Angular Y', fontsize=10, fontweight='bold')
    ax_action_ang_y.grid(True, alpha=0.3)
    ax_action_ang_y.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    
    ax_action_ang_z.set_xlabel('Phase' if phase_tensor is not None else 'Time Step')
    ax_action_ang_z.set_ylabel('Angular Z (rad/s)')
    ax_action_ang_z.set_title('Action: Angular Z', fontsize=10, fontweight='bold')
    ax_action_ang_z.grid(True, alpha=0.3)
    ax_action_ang_z.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    
    # Overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # Use tight_layout with rect to avoid overlap with suptitle
    try:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    except:
        pass  # Ignore tight_layout issues with 3D plots
    
    return fig

def parse_args():
    parser = argparse.ArgumentParser(description="Compute MMD between training and rollout datasets")
    parser.add_argument("--training_dataset", "-TDS", type=str, required=True, help="Path to training dataset (HDF5)")
    parser.add_argument("--rollout_dataset", "-RDS", type=str, required=True, help="Path to rollout dataset (HDF5)")
    parser.add_argument("--w_sigma", type=float, default=0.1, help="Weight for bandwidth in MMD kernel (default 0.1)")
    parser.add_argument("--osc_dt", type=float, default=0.02, help="Time delta for scaling actions to twists (default 0.02s for 50Hz OSC)")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    args.training_dataset = "/app/robomimic/datasets/lift/lift_feats.hdf5"
    args.rollout_dataset = "/app/robomimic/eval_data/mmd_assessments/transformer_no_divergence/lift/F_500_20/epoch_100/predicted_actions.hdf5"

    mmd_per_timestep_np, t_action_tensor, r_action_tensor, t_phase_tensor, \
        r_phase_tensor, t_demo_tensor_keys, r_demo_tensor_keys, t_ee_state_tensor, \
        r_ee_state_tensor, t_nan_mask, r_nan_mask, t_dphase, r_dphase = compute_rollout_mmd(args)

    # Visualize multiple rollout trajectories on the same plot
    num_rollouts_to_plot = min(5, r_ee_state_tensor.shape[0])  # Plot up to 5 rollouts
    rollout_indices = list(range(num_rollouts_to_plot))
    
    print(f"\nGenerating visualization for {num_rollouts_to_plot} rollouts...")
    fig = visualize_mmd_for_trajectories(
        training_ee_state_tensor=t_ee_state_tensor,
        rollout_ee_state_tensor=r_ee_state_tensor,
        training_actions_tensor=t_action_tensor,
        rollout_actions_tensor=r_action_tensor,
        mmd_tensor=mmd_per_timestep_np,
        rollout_demo_indices=rollout_indices,
        demo_keys=t_demo_tensor_keys,
        nan_mask=t_nan_mask,
        phase_tensor=t_phase_tensor,
        frame_skip=25,  # Reduced arrow frequency
        demo_indices=None,  # Use all training demos
        figsize=(24, 14),
        title=f"MMD Trajectory Visualization: Training vs {num_rollouts_to_plot} Rollouts"
    )
    
    # Save the figure
    dataset_dir = os.path.dirname(args.rollout_dataset)
    fig_path = os.path.join(dataset_dir, f"mmd_visualization_{num_rollouts_to_plot}_rollouts.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {fig_path}")
    plt.close(fig)