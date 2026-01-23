#!/usr/bin/env python3
"""
Plot success rates for transformer models trained on the lift dataset.
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


def collect_transformer_lift_data(base_dir, dataset_size, has_divergence):
    """
    Collect success rates for transformer models on lift dataset.
    
    Args:
        base_dir: Base directory containing eval data
        dataset_size: 'F' (Full), 'H1' (Half), or 'Q1' (Quarter)
        has_divergence: True for divergence models, False for no divergence
    
    Returns:
        Dictionary mapping epoch number to success rate
    """
    # Determine which subdirectory and suffix to use
    subdir = 'divergence' if has_divergence else 'no_divergence'
    suffix = 'div' if has_divergence else 'nodiv'
    
    # Target epochs: multiples of 20 up to 200
    epochs = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    
    success_rates = {}
    
    for epoch in epochs:
        # Construct filename
        filename = f"transformer_lift_{dataset_size}_500_20_{suffix}_epoch{epoch}_seed0_stats.json"
        filepath = os.path.join(base_dir, subdir, filename)
        
        # Get success rate
        success_rate = get_success_rate(filepath)
        if success_rate is not None:
            success_rates[epoch] = success_rate
    
    return success_rates


def plot_success_rates(eval_data_dir, output_file='transformer_lift_success_plot.png'):
    """
    Create plot comparing transformer model success rates on lift dataset.
    
    Args:
        eval_data_dir: Directory containing divergence and no_divergence subdirectories
        output_file: Output filename for the plot
    """
    # Dataset sizes to compare
    dataset_configs = [
        ('F', 'Full'),
        ('H1', 'Half'),
        ('Q1', 'Quarter')
    ]
    
    # Colors for each dataset size
    colors = {
        'F': '#1f77b4',      # blue
        'H1': '#ff7f0e',     # orange
        'Q1': '#2ca02c'      # green
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each configuration
    for dataset_code, dataset_label in dataset_configs:
        color = colors[dataset_code]
        
        # With divergence (solid line)
        data_div = collect_transformer_lift_data(eval_data_dir, dataset_code, has_divergence=True)
        if data_div:
            epochs = sorted(data_div.keys())
            success_rates = [data_div[e] * 100 for e in epochs]  # Convert to percentage
            ax.plot(epochs, success_rates, 
                   color=color, linestyle='-', linewidth=2,
                   label=f'{dataset_label} (with divergence)')
        
        # Without divergence (dashed line)
        data_nodiv = collect_transformer_lift_data(eval_data_dir, dataset_code, has_divergence=False)
        if data_nodiv:
            epochs = sorted(data_nodiv.keys())
            success_rates = [data_nodiv[e] * 100 for e in epochs]  # Convert to percentage
            ax.plot(epochs, success_rates, 
                   color=color, linestyle='--', linewidth=2,
                   label=f'{dataset_label} (no divergence)')
    
    # Customize plot
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Transformer Model Success Rates on Lift Dataset', 
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
    output_file = os.path.join(script_dir, 'transformer_lift_success_plot.png')
    plot_success_rates(eval_data_dir, output_file)
