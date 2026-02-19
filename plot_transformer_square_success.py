#!/usr/bin/env python3
"""
Plot success rates for transformer models trained on the square dataset.
Compares performance across different dataset sizes (Full, Half, Quarter) 
and training methods (with divergence vs. without divergence).
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def get_success_rate(stats_file):
    """Read success rate from a stats JSON file."""
    if not os.path.exists(stats_file):
        return None
    
    with open(stats_file, 'r') as f:
        data = json.load(f)
    return data.get('Success_Rate', None)


def collect_transformer_square_data(base_dir, dataset_size, has_divergence, max_seeds=10, save_frequency=20):
    """
    Collect success rates for transformer models on square dataset across multiple seeds.
    
    Args:
        base_dir: Base directory containing eval data
        dataset_size: 'F' (Full), 'H1' (Half), or 'Q1' (Quarter)
        has_divergence: True for divergence models, False for no divergence
        max_seeds: Maximum number of seeds to check
        save_frequency: Save frequency used during training (20 or 5)
    
    Returns:
        Tuple of (mean_success_rates, std_success_rates) dictionaries mapping epoch to values
    """
    # Determine which subdirectory and suffix to use
    subdir = 'divergence' if has_divergence else 'no_divergence'
    suffix = 'div' if has_divergence else 'nodiv'
    
    # Target epochs: multiples of 20 up to 200
    epochs = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    
    # Collect data across all seeds
    epoch_data = {epoch: [] for epoch in epochs}
    
    for epoch in epochs:
        for seed in range(max_seeds):
            # Construct filename
            filename = f"transformer_square_{dataset_size}_1000_{save_frequency}_{suffix}_img_epoch{epoch}_seed{seed}_stats.json"
            filepath = os.path.join(base_dir, subdir, filename)
            
            # Get success rate
            success_rate = get_success_rate(filepath)
            if success_rate is not None:
                epoch_data[epoch].append(success_rate)
    
    # Calculate mean and std for each epoch
    mean_success_rates = {}
    std_success_rates = {}
    
    for epoch in epochs:
        if epoch_data[epoch]:  # If we have data for this epoch
            mean_success_rates[epoch] = np.mean(epoch_data[epoch])
            std_success_rates[epoch] = np.std(epoch_data[epoch])
    
    return mean_success_rates, std_success_rates


def plot_success_rates(eval_data_dir, output_file='transformer_square_success_plot.png'):
    """
    Create plot comparing transformer model success rates on square dataset.
    
    Args:
        eval_data_dir: Directory containing divergence and no_divergence subdirectories
        output_file: Output filename for the plot
    """
    # Dataset sizes to compare
    dataset_configs = [
        ('F', 'Full', 20),
        ('F', 'Full (freq 5)', 5),
        ('H1', 'Half', 20),
        ('Q1', 'Quarter', 20)
    ]
    
    # Colors for each dataset size
    colors = {
        ('F', 20): '#1f77b4',      # blue
        ('F', 5): '#1f77b4',       # blue (same as freq 20)
        ('H1', 20): '#ff7f0e',     # orange
        ('Q1', 20): '#2ca02c'      # green
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each configuration
    for dataset_code, dataset_label, save_freq in dataset_configs:
        color = colors[(dataset_code, save_freq)]
        
        # With divergence (solid line)
        mean_div, std_div = collect_transformer_square_data(eval_data_dir, dataset_code, has_divergence=True, save_frequency=save_freq)
        if mean_div:
            epochs = sorted(mean_div.keys())
            success_rates = [mean_div[e] * 100 for e in epochs]  # Convert to percentage
            std_values = [std_div[e] * 100 for e in epochs]
            
            # Plot line
            line = ax.plot(epochs, success_rates, 
                   color=color, linestyle='-', linewidth=2,
                   label=f'{dataset_label} (with divergence)')
            
            # Add shaded region for standard deviation
            ax.fill_between(epochs, 
                           [sr - std for sr, std in zip(success_rates, std_values)],
                           [sr + std for sr, std in zip(success_rates, std_values)],
                           color=color, alpha=0.2)
        
        # Without divergence (dashed line) - only plot for save_freq 20 to avoid duplication
        if save_freq == 20:
            mean_nodiv, std_nodiv = collect_transformer_square_data(eval_data_dir, dataset_code, has_divergence=False, save_frequency=save_freq)
            if mean_nodiv:
                epochs = sorted(mean_nodiv.keys())
                success_rates = [mean_nodiv[e] * 100 for e in epochs]  # Convert to percentage
                std_values = [std_nodiv[e] * 100 for e in epochs]
                
                # Plot line
                ax.plot(epochs, success_rates, 
                       color=color, linestyle='--', linewidth=2,
                       label=f'{dataset_label} (no divergence)')
                
                # Add shaded region for standard deviation
                ax.fill_between(epochs, 
                               [sr - std for sr, std in zip(success_rates, std_values)],
                               [sr + std for sr, std in zip(success_rates, std_values)],
                               color=color, alpha=0.1)
    
    # Customize plot
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Transformer Model Success Rates on Square Dataset', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=True, shadow=True)
    
    # Set x-axis to show all epochs
    ax.set_xticks([20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
    
    # Set y-axis limits
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Also display
    plt.show()


if __name__ == '__main__':
    # Set path to eval_data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_data_dir = os.path.join(script_dir, 'eval_data')
    
    # Create the plot
    output_file = os.path.join(script_dir, 'transformer_square_success_plot.png')
    plot_success_rates(eval_data_dir, output_file)