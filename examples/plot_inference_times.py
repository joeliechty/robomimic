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
            epoch = part.replace("epochepoch", "")
        elif part.startswith("seed"):
            seed = part.replace("seed", "")
    
    return model, task, exp, epoch, seed

def plot_inference_times(eval_dir):
    """
    Plot mean inference times across different models and seeds.
    """
    # Find all stats JSON files
    json_files = glob.glob(os.path.join(eval_dir, "*_stats.json"))
    
    if not json_files:
        print(f"No stats files found in {eval_dir}")
        return
    
    print(f"Found {len(json_files)} evaluation files.")
    
    # Group data by model type
    data = defaultdict(lambda: {"seeds": [], "times": [], "files": []})
    
    for json_file in json_files:
        try:
            model, task, exp, epoch, seed = parse_eval_filename(json_file)
            
            with open(json_file, 'r') as f:
                stats = json.load(f)
            
            if "Inference_Time_Mean" in stats:
                # Create a label for this model variant
                label = f"{model}_{exp}"
                
                data[label]["seeds"].append(int(seed))
                data[label]["times"].append(stats["Inference_Time_Mean"])
                data[label]["files"].append(os.path.basename(json_file))
                
                print(f"{label} seed{seed}: {stats['Inference_Time_Mean']:.6f}s")
        except Exception as e:
            print(f"Error parsing {json_file}: {e}")
    
    if not data:
        print("No inference time data found.")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort models for consistent ordering
    sorted_models = sorted(data.keys())
    
    # Prepare data for plotting
    x_positions = []
    y_values = []
    colors = []
    labels_list = []
    
    # Color map for different model types
    color_map = {
        "diffusion": "blue",
        "transformer": "green",
        "vae": "red"
    }
    
    model_means = {}
    
    for i, model_label in enumerate(sorted_models):
        model_data = data[model_label]
        
        # Determine base model type for coloring
        base_model = model_label.split("_")[0]
        color = color_map.get(base_model, "gray")
        
        # Plot individual seed points
        for seed, time in zip(model_data["seeds"], model_data["times"]):
            x_positions.append(i)
            y_values.append(time)
            colors.append(color)
            labels_list.append(model_label)
        
        # Calculate mean for this model
        mean_time = np.mean(model_data["times"])
        model_means[model_label] = mean_time
        
        # Plot mean as a horizontal line
        ax.plot([i - 0.3, i + 0.3], [mean_time, mean_time], 
                color='black', linewidth=2, zorder=10)
    
    # Scatter plot of individual seeds
    ax.scatter(x_positions, y_values, c=colors, s=100, alpha=0.6, zorder=5)
    
    # Customize plot
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Inference Time (seconds)", fontsize=12)
    ax.set_title("Mean Inference Time Comparison Across Models and Seeds", fontsize=14)
    ax.set_xticks(range(len(sorted_models)))
    ax.set_xticklabels(sorted_models, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.6, label='Diffusion'),
        Patch(facecolor='green', alpha=0.6, label='Transformer'),
        Patch(facecolor='red', alpha=0.6, label='VAE'),
        plt.Line2D([0], [0], color='black', linewidth=2, label='Mean across seeds')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    
    # Save plot
    output_path = "inference_times_comparison.png"
    plt.savefig(output_path)
    print(f"\nPlot saved to {os.path.abspath(output_path)}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for model_label in sorted_models:
        model_data = data[model_label]
        times = model_data["times"]
        print(f"\n{model_label}:")
        print(f"  Mean: {np.mean(times):.6f}s")
        print(f"  Std:  {np.std(times):.6f}s")
        print(f"  Min:  {np.min(times):.6f}s")
        print(f"  Max:  {np.max(times):.6f}s")
        print(f"  Seeds: {sorted(model_data['seeds'])}")
    
    plt.show()

if __name__ == "__main__":
    # Point to eval_data directory
    EVAL_DIR = os.path.abspath("eval_data")
    
    if not os.path.exists(EVAL_DIR):
        EVAL_DIR = "/home/joe/git_repos/robomimic/eval_data"
    
    if os.path.exists(EVAL_DIR):
        plot_inference_times(EVAL_DIR)
    else:
        print(f"Evaluation directory not found: {EVAL_DIR}")
