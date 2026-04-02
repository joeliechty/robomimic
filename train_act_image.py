import os
import sys
import shutil
import tempfile
import atexit
import signal
import getpass
import argparse

import h5py
import torch

from robomimic.config import config_factory
from robomimic.scripts.train import train


def sync_all_attributes(source_path, target_path):
    """
    Copy /data-level and per-demo attributes so derived feature datasets retain metadata.
    """
    print(f"Syncing attributes from {source_path} to {target_path}...")
    with h5py.File(source_path, "r") as f_src, h5py.File(target_path, "a") as f_tgt:
        if "data" in f_src and "data" in f_tgt:
            for key, value in f_src["data"].attrs.items():
                f_tgt["data"].attrs[key] = value
            print("  [OK] Global 'data' attributes synced.")

        demos = [k for k in f_src["data"].keys() if k.startswith("demo_")]
        for demo in demos:
            if demo in f_tgt["data"]:
                for key, value in f_src[f"data/{demo}"].attrs.items():
                    f_tgt[f"data/{demo}"].attrs[key] = value
            else:
                print(f"  [Warning] {demo} found in source but not in target. Skipping.")

        print(f"  [OK] Attributes for {len(demos)} demos synced.")


_LOCAL_DATASET_COPY_PATH = None
_KEEP_LOCAL_COPY_FLAG = False
_LOCAL_COPY_CLEANUP_DONE = False


def _remove_local_dataset_copy():
    global _LOCAL_COPY_CLEANUP_DONE
    if _KEEP_LOCAL_COPY_FLAG or _LOCAL_COPY_CLEANUP_DONE:
        return
    local_path = _LOCAL_DATASET_COPY_PATH
    if local_path and os.path.isfile(local_path):
        try:
            os.remove(local_path)
            print(f"Removed temporary dataset copy: {local_path}")
        except OSError as err:
            print(f"Warning: could not remove temporary dataset copy {local_path}: {err}")
    _LOCAL_COPY_CLEANUP_DONE = True


def register_local_dataset_cleanup(local_path, keep):
    global _LOCAL_DATASET_COPY_PATH, _KEEP_LOCAL_COPY_FLAG
    _LOCAL_DATASET_COPY_PATH = local_path
    _KEEP_LOCAL_COPY_FLAG = keep
    if keep:
        print(f"Keeping local dataset copy (--keep_local_copy): {local_path}")
        return

    atexit.register(_remove_local_dataset_copy)

    def _signal_cleanup(signum, _frame):
        _remove_local_dataset_copy()
        sys.exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _signal_cleanup)
        except ValueError:
            pass


def get_local_copy_base_dir(args):
    if args.local_copy_dir:
        return os.path.abspath(os.path.expanduser(args.local_copy_dir))

    slurm_tmp = os.environ.get("SLURM_TMPDIR")
    if slurm_tmp:
        return os.path.abspath(slurm_tmp)

    user = os.environ.get("USER") or getpass.getuser()
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    scratch_local = os.path.join("/scratch/local", user, job_id)
    if os.path.isdir("/scratch/local"):
        return scratch_local

    return tempfile.gettempdir()


def make_unique_local_dataset_path(shared_target, copy_dir):
    base = os.path.basename(shared_target)
    stem, ext = os.path.splitext(base)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    pid = os.getpid()
    return os.path.join(copy_dir, f"{stem}_job{job_id}_pid{pid}{ext}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", "-D",
        type=str,
        default="lift",
        help="dataset name: 'lift', 'can', 'square', or 'tool'",
    )
    parser.add_argument(
        "--epochs", "-E",
        type=int,
        default=2000,
        help="number of training epochs",
    )
    parser.add_argument(
        "--output_dir", "-O",
        type=str,
        default="./exps/results/bc_rss/act",
        help="directory to save results",
    )
    parser.add_argument(
        "--batch_size", "-B",
        type=int,
        default=256,
        help="training batch size",
    )
    parser.add_argument(
        "--use_images", "-I",
        action="store_true",
        help="set this flag to include image feature observations",
    )
    parser.add_argument(
        "--dataset_portion", "-DP",
        type=str,
        default="full",
        choices=["full", "half", "quarter"],
        help="dataset portion: 'full', 'half', or 'quarter'",
    )
    parser.add_argument(
        "--portion_id", "-PI",
        type=int,
        default=1,
        help="which portion (1-2 for half, 1-4 for quarter, ignored for full)",
    )
    parser.add_argument(
        "--save_freq", "-SF",
        type=int,
        default=10,
        help="save checkpoint every N epochs",
    )
    parser.add_argument(
        "--end_to_end_image_training", "-E2E",
        action="store_true",
        help="train image encoders end-to-end using raw RGB observations",
    )
    parser.add_argument(
        "--validate", "-V",
        action="store_true",
        help="set this flag to run validation rollouts during training",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="set this flag to resume training from latest checkpoint",
    )
    parser.add_argument(
        "--seed", "-S",
        type=int,
        default=None,
        help="random seed for reproducibility (omit to leave unseeded)",
    )
    parser.add_argument(
        "--action_chunk_size", "-ACS",
        type=int,
        default=100,
        help="number of future actions to predict per timestep for ACT",
    )
    parser.add_argument(
        "--use_local_dataset_copy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "copy feats HDF5 to job-local storage so parallel jobs do not contend on one file "
            "(default: on). Use --no-use_local_dataset_copy to read the shared path in place."
        ),
    )
    parser.add_argument(
        "--local_copy_dir",
        type=str,
        default=None,
        help=(
            "directory for the temporary dataset copy "
            "(default: SLURM_TMPDIR, else /scratch/local/$USER/$SLURM_JOB_ID, else system temp)"
        ),
    )
    parser.add_argument(
        "--keep_local_copy",
        action="store_true",
        help="do not delete the temporary dataset copy after training",
    )
    return parser.parse_args()


def resolve_dataset_suffix(args):
    if args.dataset_portion == "full":
        return "F", ""
    if args.dataset_portion == "half":
        return f"H{args.portion_id}", f"_H{args.portion_id}"
    if args.dataset_portion == "quarter":
        return f"Q{args.portion_id}", f"_Q{args.portion_id}"
    return "F", ""


def main():
    args = parse_args()

    if args.end_to_end_image_training:
        args.use_images = False

    portion_prefix, dataset_suffix = resolve_dataset_suffix(args)

    if args.dataset not in ["lift", "can", "square", "tool"]:
        raise ValueError(
            f"Unknown dataset {args.dataset}. Please specify one of 'lift', 'can', 'square', or 'tool'."
        )

    target = f"datasets/{args.dataset}/{args.dataset}_feats{dataset_suffix}_w_cdm.hdf5"
    source = f"datasets/{args.dataset}/{args.dataset}_demo.hdf5"
    target = os.path.abspath(os.path.expanduser(target))
    source = os.path.abspath(os.path.expanduser(source))

    dataset_path = target
    if os.path.exists(source) and os.path.exists(target):
        if args.use_local_dataset_copy:
            copy_dir = get_local_copy_base_dir(args)
            os.makedirs(copy_dir, exist_ok=True)
            local_target = make_unique_local_dataset_path(target, copy_dir)
            print(
                "Per-job dataset copy (parallel-safe):\n"
                f"  shared: {target}\n"
                f"  local:  {local_target}"
            )
            try:
                shutil.copy2(target, local_target)
            except OSError as err:
                raise RuntimeError(
                    f"Failed to copy dataset to {local_target}. Check space and permissions in {copy_dir}."
                ) from err
            dataset_path = local_target
            register_local_dataset_cleanup(local_target, args.keep_local_copy)
            sync_all_attributes(source, local_target)
        else:
            sync_all_attributes(source, target)
    else:
        print("Check your file paths!")

    dataset_path = os.path.abspath(os.path.expanduser(dataset_path))
    config = config_factory(algo_name="act")

    with config.values_unlocked():
        config.train.data = dataset_path

        base_dir = args.output_dir
        if args.use_images:
            base_dir += "_images"
        elif args.end_to_end_image_training:
            base_dir += "_end2end_images"
        base_dir = os.path.join(base_dir, args.dataset)
        config.train.output_dir = base_dir

        config.observation.modalities.obs.low_dim = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object",
        ]
        if args.use_images:
            config.observation.modalities.obs.low_dim.append("robot0_eye_in_hand_feats")
            config.observation.modalities.obs.rgb = []
        elif args.end_to_end_image_training:
            config.observation.modalities.obs.rgb = ["robot0_eye_in_hand_image", "agentview_image"]
            config.observation.encoder.rgb.core_class = "VisualCore"
            config.observation.encoder.rgb.core_kwargs = {
                "backbone_class": "ResNet18Conv",
                "pool_class": "SpatialSoftmax",
                "feature_dimension": 512,
                "pretrained": False,
                "flatten": True,
            }
            config.observation.encoder.rgb.share = False
            config.observation.encoder.rgb.obs_randomizer_class = ["CropRandomizer", "ColorRandomizer"]
            config.observation.encoder.rgb.obs_randomizer_kwargs = [
                {"crop_height": 76, "crop_width": 76},
                {"brightness": 0.3, "contrast": 0.3, "saturation": 0.3, "hue": 0.1},
            ]
            config.observation.encoder.rgb.freeze = False

        config.algo.chunk_size = args.action_chunk_size
        config.train.seq_length = args.action_chunk_size
        config.train.batch_size = args.batch_size
        config.train.num_epochs = args.epochs
        config.train.cuda = torch.cuda.is_available()

        config.experiment.save.enabled = True
        config.experiment.save.every_n_epochs = args.save_freq

        if args.seed is not None:
            config.train.seed = args.seed

        seed_suffix = f"_seed{args.seed}" if args.seed is not None else ""
        chunk_suffix = f"_chunk{args.action_chunk_size}" if args.action_chunk_size > 1 else ""
        config.experiment.name = f"{portion_prefix}_{args.epochs}_{args.save_freq}{seed_suffix}{chunk_suffix}"

        if args.validate:
            config.experiment.rollout.enabled = True
            config.experiment.rollout.rate = args.save_freq
            config.experiment.rollout.n = 50
            config.experiment.rollout.horizon = 800 if args.dataset == "tool" else 400
            config.experiment.render = True
            config.experiment.render_video = True
            config.experiment.keep_all_videos = True
            if args.end_to_end_image_training:
                config.experiment.env_meta_update_dict = {
                    "env_kwargs": {
                        "has_renderer": False,
                        "has_offscreen_renderer": True,
                        "use_camera_obs": True,
                        "camera_names": ["robot0_eye_in_hand", "agentview"],
                        "camera_heights": 84,
                        "camera_widths": 84,
                    }
                }
        else:
            config.experiment.rollout.enabled = False

        config.train.hdf5_filter_key = "train"
        config.experiment.validate = True
        config.train.hdf5_validation_filter_key = "valid"

    print("Training Configuration:")
    print(config)
    train(config, device="cuda" if torch.cuda.is_available() else "cpu", resume=args.resume)


if __name__ == "__main__":
    main()
