import re
import json
import argparse
from pathlib import Path


def extract_training_times(log_file_path):
    """
    Extract epoch numbers and their corresponding wall clock training times from a log file.
    
    Args:
        log_file_path: Path to the log.txt file
        
    Returns:
        List of tuples: [(epoch_num, time_in_minutes), ...]
    """
    epoch_times = []
    
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match "Train Epoch X" followed by the JSON dict containing Time_Epoch
    # We look for the pattern across multiple lines
    pattern = r'Train Epoch (\d+)\s*\{([^}]+)\}'
    
    matches = re.finditer(pattern, content)
    
    for match in matches:
        epoch_num = int(match.group(1))
        json_content = '{' + match.group(2) + '}'
        
        try:
            # Parse the JSON to extract Time_Epoch
            epoch_data = json.loads(json_content)
            if 'Time_Epoch' in epoch_data:
                time_epoch = epoch_data['Time_Epoch']
                epoch_times.append((epoch_num, time_epoch))
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON for epoch {epoch_num}")
            continue
    
    return epoch_times


def format_time(minutes):
    """Format minutes into human-readable string"""
    seconds = minutes * 60
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif minutes < 60:
        return f"{minutes:.2f}m"
    else:
        hours = minutes / 60
        return f"{hours:.2f}h"


def process_single_log(log_file_path, output_path=None, cumulative=False, verbose=True):
    """Process a single log file and optionally save results"""
    # Extract training times
    epoch_times = extract_training_times(log_file_path)
    
    if not epoch_times:
        if verbose:
            print(f"No epoch timing information found in {log_file_path}")
        return None
    
    # Print results if verbose
    if verbose:
        print(f"\nExtracted {len(epoch_times)} epochs from {log_file_path}")
        print("-" * 70)
        
        if cumulative:
            print(f"{'Epoch':<10} {'Epoch Time':<20} {'Cumulative Time':<20}")
            print("-" * 70)
            cumulative_time = 0
            for epoch, time_min in epoch_times:
                cumulative_time += time_min
                print(f"{epoch:<10} {format_time(time_min):<20} {format_time(cumulative_time):<20}")
        else:
            print(f"{'Epoch':<10} {'Time (minutes)':<20} {'Time (formatted)':<20}")
            print("-" * 70)
            for epoch, time_min in epoch_times:
                print(f"{epoch:<10} {time_min:<20.4f} {format_time(time_min):<20}")
        
        # Calculate statistics
        times = [t for _, t in epoch_times]
        avg_time = sum(times) / len(times)
        total_time = sum(times)
        
        print("-" * 70)
        print(f"Total epochs: {len(epoch_times)}")
        print(f"Average time per epoch: {format_time(avg_time)}")
        print(f"Total training time: {format_time(total_time)}")
    
    # Save to CSV if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            if cumulative:
                f.write("epoch,epoch_time_minutes,cumulative_time_minutes\n")
                cumulative_time = 0
                for epoch, time_min in epoch_times:
                    cumulative_time += time_min
                    f.write(f"{epoch},{time_min},{cumulative_time}\n")
            else:
                f.write("epoch,time_minutes\n")
                for epoch, time_min in epoch_times:
                    f.write(f"{epoch},{time_min}\n")
        if verbose:
            print(f"\nResults saved to {output_path}")
    
    return epoch_times


def process_directory(root_dir, cumulative=False):
    """Recursively find all log.txt files and process them"""
    root_path = Path(root_dir)
    log_files = list(root_path.rglob("logs/log.txt"))
    
    if not log_files:
        print(f"No log files found in {root_dir}")
        return
    
    print(f"Found {len(log_files)} log files to process")
    print("=" * 70)
    
    processed_count = 0
    for log_file in log_files:
        print(f"\nProcessing: {log_file.relative_to(root_path)}")
        
        # Determine output path (in the same logs directory)
        output_path = log_file.parent / "training_times.csv"
        
        # Process the log file
        result = process_single_log(str(log_file), str(output_path), cumulative, verbose=False)
        
        if result:
            times = [t for _, t in result]
            avg_time = sum(times) / len(times)
            total_time = sum(times)
            print(f"  ✓ {len(result)} epochs | Avg: {format_time(avg_time)} | Total: {format_time(total_time)}")
            print(f"  → Saved to {output_path.relative_to(root_path)}")
            processed_count += 1
        else:
            print(f"  ✗ No data found")
    
    print("\n" + "=" * 70)
    print(f"Processed {processed_count}/{len(log_files)} log files successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Extract training wall clock times from robomimic training logs"
    )
    parser.add_argument(
        "--log_file", "-f",
        type=str,
        default=None,
        help="Path to a single log.txt file"
    )
    parser.add_argument(
        "--directory", "-d",
        type=str,
        default=None,
        help="Root directory to recursively search for log files"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Optional output CSV file path (only used with --log_file)"
    )
    parser.add_argument(
        "--cumulative", "-c",
        action="store_true",
        help="Show cumulative training time"
    )
    
    args = parser.parse_args()
    
    # Check that either log_file or directory is provided
    if not args.log_file and not args.directory:
        parser.error("Either --log_file or --directory must be provided")
    
    if args.log_file and args.directory:
        parser.error("Cannot use both --log_file and --directory at the same time")
    
    # Process directory mode
    if args.directory:
        process_directory(args.directory, args.cumulative)
    
    # Process single file mode
    else:
        process_single_log(args.log_file, args.output, args.cumulative, verbose=True)


if __name__ == "__main__":
    main()
