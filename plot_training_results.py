"""
Parses log.txt files from all results directories whose config folder name
contains "3000" and plots training losses + rollout success rate per epoch.

Metrics plotted (when available):
  - CDM Loss
  - L2 Loss
  - Total Loss
  - CDM Weight
  - Rollout Success Rate
"""

import os
import re
import json
import pathlib
import collections

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ─── Configuration ──────────────────────────────────────────────────────────────

RESULTS_ROOT = pathlib.Path(
    "/scratch/general/vast/u1421936/robomimic/robomimic/exps/results/bc_rss"
)
OUTPUT_DIR = pathlib.Path(
    "/scratch/general/vast/u1421936/robomimic/training_plots"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Helpers ────────────────────────────────────────────────────────────────────

# Matches:  Train Epoch 42
_RE_TRAIN_EPOCH = re.compile(r"^Train Epoch (\d+)\s*$", re.MULTILINE)
# Matches:  Epoch 100 Rollouts took … with results:
_RE_ROLLOUT_EPOCH = re.compile(r"^Epoch (\d+) Rollouts took", re.MULTILINE)
# JSON block (greedy from first { to matching })
_RE_JSON_BLOCK = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _find_json_after(text: str, pos: int) -> dict | None:
    """Return the first JSON dict that starts at or after `pos`."""
    m = _RE_JSON_BLOCK.search(text, pos)
    if m is None:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def parse_log(log_path: pathlib.Path) -> dict:
    """
    Returns a dict with keys:
      epochs        – list of int
      cdm_loss      – list of float | None
      l2_loss       – list of float | None
      total_loss    – list of float | None
      cdm_weight    – list of float | None
      rollout_epochs – list of int
      success_rates  – list of float
    """
    text = log_path.read_text(errors="replace")

    epochs, cdm_loss, l2_loss, total_loss, cdm_weight = [], [], [], [], []

    for m in _RE_TRAIN_EPOCH.finditer(text):
        epoch = int(m.group(1))
        data = _find_json_after(text, m.end())
        if data is None:
            continue
        epochs.append(epoch)
        cdm_loss.append(data.get("CDM_Loss"))
        l2_loss.append(data.get("L2_Loss"))
        total_loss.append(data.get("Loss"))
        cdm_weight.append(data.get("CDM_Weight"))

    rollout_epochs, success_rates = [], []
    for m in _RE_ROLLOUT_EPOCH.finditer(text):
        epoch = int(m.group(1))
        # The rollout results JSON comes right after the "Env: …" line
        search_start = m.end()
        data = _find_json_after(text, search_start)
        if data is None:
            continue
        sr = data.get("Success_Rate")
        if sr is not None:
            rollout_epochs.append(epoch)
            success_rates.append(sr)

    return dict(
        epochs=epochs,
        cdm_loss=cdm_loss,
        l2_loss=l2_loss,
        total_loss=total_loss,
        cdm_weight=cdm_weight,
        rollout_epochs=rollout_epochs,
        success_rates=success_rates,
    )


# ─── Discovery ──────────────────────────────────────────────────────────────────

RunInfo = collections.namedtuple(
    "RunInfo", ["model_type", "task", "config_name", "log_path", "label"]
)

def discover_runs(results_root: pathlib.Path) -> list[RunInfo]:
    """
    Walk results_root and collect every run whose config directory name
    (the level just above the timestamp) contains "3000".
    """
    runs = []
    for model_type_dir in sorted(results_root.iterdir()):
        if not model_type_dir.is_dir():
            continue
        model_type = model_type_dir.name

        for task_dir in sorted(model_type_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task = task_dir.name

            for config_dir in sorted(task_dir.iterdir()):
                if not config_dir.is_dir():
                    continue
                if "3000" not in config_dir.name:
                    continue

                # Pick the most recent timestamped sub-directory
                ts_dirs = sorted(
                    [d for d in config_dir.iterdir() if d.is_dir()],
                    key=lambda d: d.name,
                )
                if not ts_dirs:
                    continue
                run_dir = ts_dirs[-1]

                log_path = run_dir / "logs" / "log.txt"
                if not log_path.exists():
                    continue

                short_model = (
                    model_type
                    .replace("_end2end_images", "")
                    .replace("transformer_no_divergence", "transformer_NoCDM")
                    .replace("transformer_divergence", "transformer_CDM")
                    .replace("diffusion_policy", "diffusion")
                )
                label = f"{short_model}/{task}/{config_dir.name}"
                runs.append(RunInfo(model_type, task, config_dir.name, log_path, label))

    return runs


# ─── Plotting ───────────────────────────────────────────────────────────────────

METRICS = [
    ("CDM Loss",       "cdm_loss",    "epochs"),
    ("L2 Loss",        "l2_loss",     "epochs"),
    ("Total Loss",     "total_loss",  "epochs"),
    ("CDM Weight",     "cdm_weight",  "epochs"),
    ("Success Rate",   "success_rates", "rollout_epochs"),
]


def _values_present(data: dict, val_key: str) -> bool:
    return any(v is not None for v in data[val_key])


def plot_task(task: str, runs: list[tuple[RunInfo, dict]], out_dir: pathlib.Path):
    """Create one figure per task with subplots for each available metric."""

    # Decide which metrics are non-empty across all runs for this task
    active_metrics = []
    for name, val_key, epoch_key in METRICS:
        for _, data in runs:
            if _values_present(data, val_key):
                active_metrics.append((name, val_key, epoch_key))
                break

    if not active_metrics:
        print(f"  No data for task {task}, skipping.")
        return

    n_metrics = len(active_metrics)
    fig, axes = plt.subplots(
        1, n_metrics, figsize=(5 * n_metrics, 5), squeeze=False
    )
    axes = axes[0]

    colors = cm.tab10(np.linspace(0, 1, max(len(runs), 1)))

    for ax, (metric_name, val_key, epoch_key) in zip(axes, active_metrics):
        for (run_info, data), color in zip(runs, colors):
            xs = data[epoch_key]
            ys = data[val_key]

            # Filter out None values (keep paired xs/ys)
            pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if not pairs:
                continue
            xs_clean, ys_clean = zip(*pairs)

            ax.plot(xs_clean, ys_clean, label=run_info.label, color=color, linewidth=1.5)

        ax.set_title(metric_name)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    # Single shared legend below all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    # Collect from all axes in case some only appear in later subplots
    for ax in axes[1:]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(len(runs), 3),
        bbox_to_anchor=(0.5, -0.18),
        fontsize=8,
        framealpha=0.8,
    )
    fig.suptitle(f"Task: {task}", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path = out_dir / f"{task}_training_metrics.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ─── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Scanning: {RESULTS_ROOT}")
    runs = discover_runs(RESULTS_ROOT)
    print(f"Found {len(runs)} runs with '3000' in config name.\n")

    # Parse all logs
    parsed: list[tuple[RunInfo, dict]] = []
    for run_info in runs:
        print(f"  Parsing: {run_info.label}")
        data = parse_log(run_info.log_path)
        print(
            f"    epochs={len(data['epochs'])}, "
            f"rollout_epochs={data['rollout_epochs']}, "
            f"success_rates={data['success_rates']}"
        )
        parsed.append((run_info, data))

    # Group by task
    by_task: dict[str, list] = collections.defaultdict(list)
    for item in parsed:
        by_task[item[0].task].append(item)

    print(f"\nGenerating plots in: {OUTPUT_DIR}\n")
    for task, task_runs in sorted(by_task.items()):
        print(f"Task: {task}  ({len(task_runs)} runs)")
        plot_task(task, task_runs, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
