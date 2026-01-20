import os
import json
import matplotlib.pyplot as plt
import glob
import re
import numpy as np

def parse_log_file(log_path):
    """
    Parses a log.txt file to extract training metrics per epoch.
    Returns a dictionary of lists: {metric_name: [value_epoch_0, value_epoch_1, ...]}
    """
    data = {}
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for "Train Epoch <N>"
        if line.startswith("Train Epoch"):
            # The next lines should contain the JSON stats
            json_str = ""
            i += 1
            if i < len(lines) and lines[i].strip() == "{":
                while i < len(lines):
                    json_str += lines[i]
                    if lines[i].strip() == "}":
                        break
                    i += 1
                
                try:
                    stats = json.loads(json_str)
                    for key, value in stats.items():
                        if key not in data:
                            data[key] = []
                        data[key].append(value)
                except json.JSONDecodeError:
                    pass
        i += 1
        
    return data

def find_variable_name(data, candidates):
    """Finds the first matching key from candidates in data."""
    for candidate in candidates:
        if candidate in data:
            return candidate
    return None

def plot_losses(root_dir):
    # Find all log.txt files
    log_files = glob.glob(os.path.join(root_dir, "**", "log.txt"), recursive=True)
    
    if not log_files:
        print(f"No log.txt files found in {root_dir}")
        return

    print(f"Found {len(log_files)} log files.")
    
    # Store data for plotting
    experiments = {}

    for log_file in log_files:
        # Infer meaningful name from path
        # Path structure: .../results/bc_rss/<model_type>/<dataset>/<exp_name>/<date>/logs/log.txt
        parts = log_file.split(os.sep)
        try:
            # Adjust index based on your specific folder structure
            # Assuming structure: .../bc_rss/model_type/dataset/exp_name/...
            idx = parts.index("bc_rss")
            model_type = parts[idx+1]
            exp_name = parts[idx+3] # exp0, exp1, etc
            run_id = parts[idx+4]   # date/timestamp
            
            label = f"{model_type} ({exp_name})"
        except (ValueError, IndexError):
            label = os.path.dirname(log_file)[-20:] # Fallback

        print(f"Parsing: {label} -> {log_file}")
        data = parse_log_file(log_file)
        if data:
            experiments[label] = data

    # Setup plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Create consistent color mapping for all experiments
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    color_map = {label: colors[i] for i, label in enumerate(experiments.keys())}
    
    # Plot 1: CDM Loss
    ax = axes[0]
    has_cdm = False
    for label, data in experiments.items():
        if "CDM_Loss" in data:
            epochs = range(1, len(data["CDM_Loss"]) + 1)
            ax.plot(epochs, data["CDM_Loss"], label=label, color=color_map[label])
            has_cdm = True
            
    if has_cdm:
        ax.set_title("CDM Loss over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("CDM Loss")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
    else:
        ax.set_title("No CDM Loss found in logs")

    # Plot 2: Reconstruction / L2 Loss
    ax = axes[1]
    # different models name this differently
    recon_keys = ["Reconstruction_Loss", "L2_Loss", "Loss"] 
    
    for label, data in experiments.items():
        key = find_variable_name(data, recon_keys)
        
        # Special case: For diffusion, "Loss" is effectively the reconstruction (denoising) loss.
        # But allow it to be plotted alongside L2/Recon from others.
        
        if key:
            epochs = range(1, len(data[key]) + 1)
            ax.plot(epochs, data[key], label=f"{label} ({key})", color=color_map[label])

    ax.set_title("Reconstruction / L2 Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)
    
    loss_plot_path = "training_losses_comparison.png"
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    print(f"\nPlot saved to {os.path.abspath(loss_plot_path)}")
    plt.show()

if __name__ == "__main__":
    # Point this to your results folder
    RESULTS_DIR = os.path.abspath("robomimic/exps/results/bc_rss")
    
    if not os.path.exists(RESULTS_DIR):
        # Fallback to current directory or search
        # Trying the path observed in your workspace
        RESULTS_DIR = "/home/joe/git_repos/robomimic/robomimic/exps/results/bc_rss"

    plot_losses(RESULTS_DIR)
