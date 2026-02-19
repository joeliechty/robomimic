import os
import glob
import subprocess
import argparse
import h5py
from datetime import datetime

def check_has_obs(hdf5_path):
    """Check if an HDF5 file already has observations."""
    try:
        with h5py.File(hdf5_path, "r") as f:
            demos = list(f["data"].keys())
            if len(demos) > 0:
                demo_key = demos[0]
                return "obs" in f[f"data/{demo_key}"]
    except Exception as e:
        print(f"Warning: Could not check {hdf5_path}: {e}")
        return False
    return False

def cleanup_leftover_fixed_files(base_dir):
    """Clean up any leftover _fixed.hdf5 files from previous interrupted runs."""
    fixed_files = glob.glob(os.path.join(base_dir, "**", "*_fixed.hdf5"), recursive=True)
    if fixed_files:
        print(f"\n⚠ Found {len(fixed_files)} leftover _fixed.hdf5 files from previous runs:")
        for f in fixed_files:
            print(f"  - {f}")
        response = input("\nDelete these leftover files? (y/n): ")
        if response.lower() == 'y':
            for f in fixed_files:
                os.remove(f)
                print(f"  Deleted: {f}")
            print("✓ Cleanup complete\n")
        else:
            print("Skipping cleanup\n")

def fix_rollouts(base_dir, dry_run=False, skip_existing=True):
    """Fix rollout HDF5 files by adding observations from states.
    
    Args:
        base_dir: Base directory to search for HDF5 files
        dry_run: If True, only show what would be done without making changes
        skip_existing: If True, skip files that already have observations
    """
    # Clean up leftover files first
    if not dry_run:
        cleanup_leftover_fixed_files(base_dir)
    
    # Find all hdf5 files recursively (excluding _fixed files)
    all_files = glob.glob(os.path.join(base_dir, "**", "*.hdf5"), recursive=True)
    files = [f for f in all_files if not f.endswith("_fixed.hdf5")]
    
    # Locate the dataset_states_to_obs.py script
    script_path = os.path.join("robomimic", "scripts", "dataset_states_to_obs.py")
    if not os.path.exists(script_path):
        print(f"Error: Could not find {script_path}. Please run this script from the root of the robomimic repository.")
        return

    print(f"\n{'=' * 80}")
    print(f"Found {len(files)} hdf5 files in {base_dir}")
    print(f"Mode: {'DRY RUN (no changes will be made)' if dry_run else 'LIVE (will modify files)'}")
    print(f"Skip existing: {skip_existing}")
    print(f"{'=' * 80}\n")
    
    # Filter files that need processing
    files_to_process = []
    files_skipped = []
    
    print("Checking which files need processing...")
    for file_path in files:
        if skip_existing and check_has_obs(file_path):
            files_skipped.append(file_path)
        else:
            files_to_process.append(file_path)
    
    print(f"\n✓ Files to process: {len(files_to_process)}")
    print(f"✓ Files to skip (already have obs): {len(files_skipped)}\n")
    
    if files_skipped:
        print(f"Skipping {len(files_skipped)} files that already have observations")
        if len(files_skipped) <= 10:
            for f in files_skipped:
                print(f"  - {os.path.relpath(f, base_dir)}")
        else:
            print(f"  (showing first 5)")
            for f in files_skipped[:5]:
                print(f"  - {os.path.relpath(f, base_dir)}")
        print()
    
    if not files_to_process:
        print("✓ All files already have observations! Nothing to do.")
        return
    
    if dry_run:
        print("DRY RUN - Would process these files:")
        for f in files_to_process:
            print(f"  - {os.path.relpath(f, base_dir)}")
        print("\nRun without --dry_run to actually process files.")
        return
    
    # Confirm before proceeding
    print(f"About to process {len(files_to_process)} files.")
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    print(f"\n{'=' * 80}")
    print(f"Starting processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n")
    
    successful = []
    failed = []
    
    for i, file_path in enumerate(files_to_process):
        rel_path = os.path.relpath(file_path, base_dir)
        print(f"\n{'=' * 80}")
        print(f"[{i+1}/{len(files_to_process)}] Processing: {rel_path}")
        print(f"{'=' * 80}")
        
        # Create a temporary output path
        temp_output_name = os.path.basename(file_path).replace(".hdf5", "_fixed.hdf5")
        generated_path = os.path.join(os.path.dirname(file_path), temp_output_name)
        
        # Construct command
        # done_mode=2 ensures we capture success/failure signals correctly
        # We omit camera arguments to generate low-dim observations only
        cmd = [
            "python", script_path,
            "--dataset", file_path,
            "--output_name", temp_output_name,
            "--done_mode", "2"
        ]
        
        try:
            # Run the conversion script
            print(f"Running: {' '.join(cmd)}\n")
            subprocess.check_call(cmd)
            
            if os.path.exists(generated_path):
                # Verify the fixed file has observations
                if check_has_obs(generated_path):
                    # Replace the original file with the fixed one
                    print(f"\n✓ Replacing original file with fixed version...")
                    os.replace(generated_path, file_path)
                    print(f"✓ Successfully fixed: {rel_path}")
                    successful.append(rel_path)
                else:
                    print(f"⚠ Warning: Fixed file {generated_path} still missing observations!")
                    print(f"  Keeping both files for manual inspection.")
                    failed.append(rel_path)
            else:
                print(f"✗ Error: Expected output file {generated_path} not found.")
                failed.append(rel_path)

        except subprocess.CalledProcessError as e:
            print(f"\n✗ Failed to process {rel_path}: {e}")
            failed.append(rel_path)
            # Cleanup if needed
            if os.path.exists(generated_path):
                print(f"  Cleaning up temporary file: {generated_path}")
                os.remove(generated_path)
        except KeyboardInterrupt:
            print(f"\n\n⚠ Interrupted by user!")
            if os.path.exists(generated_path):
                print(f"⚠ Temporary file left at: {generated_path}")
                print(f"  You can manually inspect it or delete it.")
            print(f"\nProgress so far:")
            print(f"  Successful: {len(successful)}")
            print(f"  Failed: {len(failed)}")
            print(f"  Remaining: {len(files_to_process) - i - 1}")
            raise
        except Exception as e:
            print(f"\n✗ An error occurred: {e}")
            failed.append(rel_path)
    
    # Summary
    print(f"\n\n{'=' * 80}")
    print(f"PROCESSING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total files processed: {len(files_to_process)}")
    print(f"✓ Successful: {len(successful)}")
    print(f"✗ Failed: {len(failed)}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n")
    
    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  - {f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix rollout HDF5 files by adding observations from states")
    parser.add_argument("--dir", type=str, default="eval_data", help="Directory to search for hdf5 files")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--process_all", action="store_true", help="Process all files, even those that already have observations")
    args = parser.parse_args()
    
    fix_rollouts(args.dir, dry_run=args.dry_run, skip_existing=not args.process_all)