#!/usr/bin/env python
"""
Script to inspect the contents of rollout_data.hdf5
"""
import h5py
import numpy as np
import json

def print_hdf5_structure(name, obj, indent=0):
    """Recursively print HDF5 file structure"""
    prefix = "  " * indent
    if isinstance(obj, h5py.Group):
        print(f"{prefix}📁 Group: {name}")
        print(f"{prefix}   Attributes: {dict(obj.attrs)}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{prefix}📄 Dataset: {name}")
        print(f"{prefix}   Shape: {obj.shape}")
        print(f"{prefix}   Dtype: {obj.dtype}")
        print(f"{prefix}   Attributes: {dict(obj.attrs)}")
        # Show a preview of the data if it's small enough
        if obj.size < 100 and obj.size > 0:
            print(f"{prefix}   Data preview: {obj[...]}")
        elif obj.size > 0:
            print(f"{prefix}   Data preview (first few): {obj.flat[:min(10, obj.size)]}")

def main():
    hdf5_path = "eval_data/no_divergence/diffusion_square_F_1000_5_nodiv_img_epoch250_seed0.hdf5"
    
    print(f"=" * 80)
    print(f"Inspecting: {hdf5_path}")
    print(f"=" * 80)
    print()
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            print("🔍 ROOT LEVEL ATTRIBUTES:")
            for key, value in f.attrs.items():
                print(f"  {key}: {value}")
            print()
            
            print("🔍 FILE STRUCTURE:")
            print()
            f.visititems(print_hdf5_structure)
            print()
            
            print("=" * 80)
            print("📊 SUMMARY:")
            print("=" * 80)
            
            # Count datasets and groups
            num_groups = 0
            num_datasets = 0
            dataset_names = []
            
            def count_items(name, obj):
                nonlocal num_groups, num_datasets
                if isinstance(obj, h5py.Group):
                    num_groups += 1
                elif isinstance(obj, h5py.Dataset):
                    num_datasets += 1
                    dataset_names.append(name)
            
            f.visititems(count_items)
            
            print(f"Total Groups: {num_groups}")
            print(f"Total Datasets: {num_datasets}")
            print()
            print("Dataset names:")
            for name in sorted(dataset_names):
                print(f"  - {name}")
            
    except FileNotFoundError:
        print(f"❌ Error: File not found at {hdf5_path}")
    except Exception as e:
        print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    main()
