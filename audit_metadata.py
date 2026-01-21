import json
import h5py
original_path = "dataset/can/can_demo.hdf5" # The raw dataset with pixels
feature_path = "dataset/can/can_feats.hdf5" # Your new feature dataset

def audit_metadata(path, label):
    print(f"\n--- Auditing {label}: {path} ---")
    try:
        with h5py.File(path, 'r') as f:
            if "env_args" in f["data"].attrs:
                env_args = json.loads(f["data"].attrs["env_args"])
                
                # Check top level
                cam_names = env_args.get("camera_names", None)
                print(f"Top-level 'camera_names': {cam_names} ({type(cam_names)})")
                
                # Check nested env_kwargs (often used in robosuite)
                if "env_kwargs" in env_args:
                    kwarg_cams = env_args["env_kwargs"].get("camera_names", None)
                    print(f"Nested 'env_kwargs -> camera_names': {kwarg_cams} ({type(kwarg_cams)})")
                
                # Look for ANY string that might be "agentview" instead of ["agentview"]
                for k, v in env_args.items():
                    if v == "agentview":
                        print(f"WARNING: Found raw string 'agentview' in key: {k}")
            else:
                print("Attribute 'env_args' missing!")
    except Exception as e:
        print(f"Could not audit {label}: {e}")

# Run the audit

audit_metadata(original_path, "ORIGINAL")
audit_metadata(feature_path, "FEATURE")