#!/usr/bin/env python
"""
Script to compare timing data across different tasks and dataset sizes.
Reads all *_time.json files and creates a comparison table.
"""
import os
import json
import pandas as pd
from pathlib import Path


def load_time_data():
    """Load all time JSON files from the datasets directory."""
    datasets_dir = Path(__file__).parent / "datasets"
    
    tasks = ["lift", "can", "square"]
    dataset_sizes = ["F", "H1", "H2", "Q1", "Q2", "Q3", "Q4"]
    
    # Mapping of task to filename pattern
    file_patterns = {
        "lift": "lift_feats",
        "can": "can_feats",
        "square": "square_feats"
    }
    
    # Mapping of dataset size to filename suffix
    size_suffixes = {
        "F": "",  # Full dataset has no suffix
        "H1": "_H1",
        "H2": "_H2",
        "Q1": "_Q1",
        "Q2": "_Q2",
        "Q3": "_Q3",
        "Q4": "_Q4"
    }
    
    data = []
    
    for task in tasks:
        task_dir = datasets_dir / task
        if not task_dir.exists():
            print(f"Warning: {task_dir} does not exist")
            continue
            
        for size in dataset_sizes:
            # Build filename
            filename = f"{file_patterns[task]}{size_suffixes[size]}_time.json"
            filepath = task_dir / filename
            
            if filepath.exists():
                with open(filepath, 'r') as f:
                    json_data = json.load(f)
                    
                    # Extract relevant information
                    row = {
                        "Task": task,
                        "Dataset Size": size,
                        "Time (s)": round(json_data["elapsed_time_seconds"], 3),
                        "n_demos": json_data["n_demos"],
                        "Traj Length Mean": round(json_data["trajectory_length_stats"]["mean"], 2),
                        "Traj Length Std": round(json_data["trajectory_length_stats"]["std"], 2),
                        "Traj Length Min": json_data["trajectory_length_stats"]["min"],
                        "Traj Length Max": json_data["trajectory_length_stats"]["max"],
                        "Traj Length Median": json_data["trajectory_length_stats"]["median"]
                    }
                    data.append(row)
            else:
                print(f"Warning: {filepath} not found")
    
    return pd.DataFrame(data)


def print_summary_table(df):
    """Print a summary table grouped by task."""
    print("\n" + "="*100)
    print("DATASET TIMING COMPARISON")
    print("="*100)
    
    for task in df["Task"].unique():
        task_df = df[df["Task"] == task].sort_values("Dataset Size")
        print(f"\n{'─'*100}")
        print(f"Task: {task.upper()}")
        print(f"{'─'*100}")
        print(task_df.to_string(index=False))
    
    print("\n" + "="*100)


def print_comparison_by_size(df):
    """Print comparison table organized by dataset size."""
    print("\n" + "="*100)
    print("COMPARISON BY DATASET SIZE")
    print("="*100)
    
    # Order dataset sizes logically
    size_order = ["F", "H1", "H2", "Q1", "Q2", "Q3", "Q4"]
    
    for size in size_order:
        size_df = df[df["Dataset Size"] == size]
        if not size_df.empty:
            print(f"\n{'─'*100}")
            print(f"Dataset Size: {size}")
            print(f"{'─'*100}")
            print(size_df.to_string(index=False))
    
    print("\n" + "="*100)


def print_compact_table(df):
    """Print a compact pivot table showing time and n_demos."""
    print("\n" + "="*100)
    print("COMPACT SUMMARY: Time (seconds) and Number of Demos")
    print("="*100)
    
    # Create pivot for time
    time_pivot = df.pivot(index="Dataset Size", columns="Task", values="Time (s)")
    demos_pivot = df.pivot(index="Dataset Size", columns="Task", values="n_demos")
    
    # Reorder rows
    size_order = ["F", "H1", "H2", "Q1", "Q2", "Q3", "Q4"]
    time_pivot = time_pivot.reindex([s for s in size_order if s in time_pivot.index])
    demos_pivot = demos_pivot.reindex([s for s in size_order if s in demos_pivot.index])
    
    print("\nProcessing Time (seconds):")
    print(time_pivot.to_string())
    
    print("\n\nNumber of Demonstrations:")
    print(demos_pivot.to_string())
    
    print("\n" + "="*100)


def export_to_csv(df, output_file="dataset_times_comparison.csv"):
    """Export the data to CSV."""
    df.to_csv(output_file, index=False)
    print(f"\nData exported to: {output_file}")


def main():
    # Load data
    df = load_time_data()
    
    if df.empty:
        print("No data found!")
        return
    
    # Print various views
    print_summary_table(df)
    print_comparison_by_size(df)
    print_compact_table(df)
    
    # Export to CSV
    export_to_csv(df)
    
    # Print some statistics
    print("\n" + "="*100)
    print("STATISTICS")
    print("="*100)
    print(f"\nTotal datasets analyzed: {len(df)}")
    print(f"Tasks: {', '.join(df['Task'].unique())}")
    print(f"Dataset sizes: {', '.join(sorted(df['Dataset Size'].unique()))}")
    print(f"\nTotal processing time: {df['Time (s)'].sum():.2f} seconds")
    print(f"Average processing time: {df['Time (s)'].mean():.2f} seconds")
    print(f"Total demonstrations: {df['n_demos'].sum()}")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
