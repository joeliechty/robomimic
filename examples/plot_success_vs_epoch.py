import os
import json
import matplotlib.pyplot as plt
import glob
import numpy as np
from collections import defaultdict

def parse_eval_filename(filename):
    """
    Parse evaluation filename to extract model info.
    Expected format: {model}_{task}_exp{exp_num}_epochepoch{epoch}_seed{seed}_stats.json
    Returns: (model, task, exp_num, epoch, seed)
    """
    basename = os.path.basename(filename)
    basename = basename.replace("_stats.json", "")
    
    parts = basename.split("_")
    
    # Extract components
    model = parts[0]  # diffusion, transformer, vae
    task = parts[1]   # lift, can, square
    exp = parts[2]    # exp0, exp1, etc
    
    # Find epoch and seed
    epoch = None
    seed = None
    for i, part in enumerate(parts):
        if part.startswith("epochepoch"):
            epoch = int(part.replace("epochepoch", ""))
        elif part.startswith("seed"):
            seed = int(part.replace("seed", ""))
    
    return model, task, exp, epoch, seed

def plot_success_and_horizon(eval_dir):
    """
    Plot Success Rate and Horizon vs Epoch on dual y-axes.
    """
    # Find all stats JSON files
    json_files = glob.glob(os.path.join(eval_dir, "*_stats.json"))
    
    if not json_files:
        print(f"No stats files found in {eval_dir}")
        return
    
    print(f"Found {len(json_files)} evaluation files.")
    
    # Group data by model type
    # Structure: data[model_label][epoch][seed] = {"success_rate": x, "horizon": y}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for json_file in json_files:
        try:
            model, task, exp, epoch, seed = parse_eval_filename(json_file)
            
            with open(json_file, 'r') as f:
                stats = json.load(f)
            
            if "Success_Rate" in stats and "Horizon" in stats:
                # Create a label for this model variant
                label = f"{model}_{exp}"
                
                data[label][epoch][seed] = {
                    "success_rate": stats["Success_Rate"],
                    "horizon": stats["Horizon"]
                }
                
                print(f"{label} epoch{epoch} seed{seed}: SR={stats['Success_Rate']:.2f}, H={stats['Horizon']:.1f}")
        except Exception as e:
            print(f"Error parsing {json_file}: {e}")
    
    if not data:
        print("No success rate/horizon data found.")
        return
    
    # Create plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Sort models for consistent ordering
    sorted_models = sorted(data.keys())
    
    # Color map for different model types
    color_cycle = plt.cm.tab10(np.linspace(0, 1, len(sorted_models)))
    
    # Plot on primary axis (Success Rate)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Success Rate", fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create secondary axis (Horizon)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Horizon (Steps)", fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Plot data for each model
    for i, model_label in enumerate(sorted_models):
        model_data = data[model_label]
        color = color_cycle[i]
        
        # Aggregate across seeds for each epoch
        epochs = sorted(model_data.keys())
        success_rates_mean = []
        success_rates_std = []
        horizons_mean = []
        horizons_std = []
        
        for epoch in epochs:
            seed_data = model_data[epoch]
            
            success_rates = [seed_data[seed]["success_rate"] for seed in seed_data]
            horizons = [seed_data[seed]["horizon"] for seed in seed_data]
            
            success_rates_mean.append(np.mean(success_rates))
            success_rates_std.append(np.std(success_rates))
            horizons_mean.append(np.mean(horizons))
            horizons_std.append(np.std(horizons))
        
        # Plot Success Rate on left axis (solid line)
        ax1.plot(epochs, success_rates_mean, 
                 color=color, marker='o', linestyle='-', linewidth=2,
                 label=f"{model_label} (SR)", alpha=0.8)
        ax1.fill_between(epochs, 
                         np.array(success_rates_mean) - np.array(success_rates_std),
                         np.array(success_rates_mean) + np.array(success_rates_std),
                         color=color, alpha=0.2)
        
        # Plot Horizon on right axis (dashed line)
        ax2.plot(epochs, horizons_mean, 
                 color=color, marker='s', linestyle='--', linewidth=2,
                 label=f"{model_label} (H)", alpha=0.8)
        ax2.fill_between(epochs, 
                         np.array(horizons_mean) - np.array(horizons_std),
                         np.array(horizons_mean) + np.array(horizons_std),
                         color=color, alpha=0.2)
    
    # Add title and grid
    plt.title("Success Rate and Horizon vs Epoch", fontsize=14, pad=20)
    ax1.grid(True, alpha=0.3)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Create custom legend
    legend_elements = []
    from matplotlib.lines import Line2D
    legend_elements.append(Line2D([0], [0], color='blue', linewidth=2, label='Success Rate (left axis)'))
    legend_elements.append(Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Horizon (right axis)'))
    legend_elements.append(Line2D([0], [0], color='white', linewidth=0, label=''))  # spacer
    
    for i, model_label in enumerate(sorted_models):
        color = color_cycle[i]
        legend_elements.append(Line2D([0], [0], color=color, linewidth=2, marker='o', label=model_label))
    
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_path = "success_horizon_vs_epoch.png"
    plt.savefig(output_path)
    print(f"\nPlot saved to {os.path.abspath(output_path)}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for model_label in sorted_models:
        model_data = data[model_label]
        print(f"\n{model_label}:")
        for epoch in sorted(model_data.keys()):
            seed_data = model_data[epoch]
            success_rates = [seed_data[seed]["success_rate"] for seed in seed_data]
            horizons = [seed_data[seed]["horizon"] for seed in seed_data]
            print(f"  Epoch {epoch}:")
            print(f"    Success Rate: {np.mean(success_rates):.3f} ± {np.std(success_rates):.3f}")
            print(f"    Horizon: {np.mean(horizons):.1f} ± {np.std(horizons):.1f}")
    
    plt.show()

if __name__ == "__main__":
    # Point to eval_data directory
    EVAL_DIR = os.path.abspath("eval_data")
    
    if not os.path.exists(EVAL_DIR):
        EVAL_DIR = "/home/joe/git_repos/robomimic/eval_data"
    
    if os.path.exists(EVAL_DIR):
        plot_success_and_horizon(EVAL_DIR)
    else:
        print(f"Evaluation directory not found: {EVAL_DIR}")
