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

import math

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
# Matches:  Validation Epoch 42
_RE_VAL_EPOCH = re.compile(r"^Validation Epoch (\d+)\s*$", re.MULTILINE)
# Matches:  Epoch 100 Rollouts took … with results:
_RE_ROLLOUT_EPOCH = re.compile(r"^Epoch (\d+) Rollouts took", re.MULTILINE)
# JSON block (greedy from first { to matching })
_RE_JSON_BLOCK = re.compile(r"\{[^{}]*\}", re.DOTALL)
# Strips _seedN suffix to get the base config name for grouping
_RE_SEED_SUFFIX = re.compile(r"_seed\d+$")


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
      epochs         – list of int
      cdm_loss       – list of float | None
      l2_loss        – list of float | None
      total_loss     – list of float | None
      cdm_weight     – list of float | None
      val_epochs     – list of int
      val_cdm_loss   – list of float | None
      val_l2_loss    – list of float | None
      val_total_loss – list of float | None
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

    val_epochs, val_cdm_loss, val_l2_loss, val_total_loss = [], [], [], []

    for m in _RE_VAL_EPOCH.finditer(text):
        epoch = int(m.group(1))
        data = _find_json_after(text, m.end())
        if data is None:
            continue
        val_epochs.append(epoch)
        val_cdm_loss.append(data.get("CDM_Loss"))
        val_l2_loss.append(data.get("L2_Loss"))
        val_total_loss.append(data.get("Loss"))

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
        val_epochs=val_epochs,
        val_cdm_loss=val_cdm_loss,
        val_l2_loss=val_l2_loss,
        val_total_loss=val_total_loss,
        rollout_epochs=rollout_epochs,
        success_rates=success_rates,
    )


# ─── Discovery ──────────────────────────────────────────────────────────────────

RunInfo = collections.namedtuple(
    "RunInfo", ["model_type", "task", "config_name", "base_config_name", "log_path", "label"]
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

            # Per-task filter for config directory names
            TASK_FILTERS = {
                "can": ["3000_50", "1000_50"],
                "lift": ["50_1"],
                "square": ["4000_100"]
            }
            task_filter = TASK_FILTERS.get(task)

            for config_dir in sorted(task_dir.iterdir()):
                if not config_dir.is_dir():
                    continue
                if task_filter is not None and not any(f in config_dir.name for f in task_filter):
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
                base_config = _RE_SEED_SUFFIX.sub("", config_dir.name)
                label = f"{short_model}/{task}/{base_config}"
                runs.append(RunInfo(model_type, task, config_dir.name, base_config, log_path, label))

    return runs


# ─── Plotting ───────────────────────────────────────────────────────────────────

# Each entry: (title, use_log_scale, series_list)
# series_list items: (val_key, epoch_key, linestyle, label_suffix)
# label_suffix is appended to the run label; use "" for single-series subplots.
METRICS = [
    ("Train CDM & L2 Loss", True, [
        ("cdm_loss", "epochs", "--", " (CDM)"),
        ("l2_loss",  "epochs", "-",  " (L2)"),
    ]),
    ("Val CDM & L2 Loss", True, [
        ("val_cdm_loss", "val_epochs", "--", " (CDM)"),
        ("val_l2_loss",  "val_epochs", "-",  " (L2)"),
    ]),
    ("Train Total Loss", True,  [("total_loss",     "epochs",         "-", "")]),
    ("Val Total Loss",   True,  [("val_total_loss",  "val_epochs",     "-", "")]),
    ("CDM Weight",       True,  [("cdm_weight",      "epochs",         "-", "")]),
    ("Success Rate",     False, [("success_rates",   "rollout_epochs", "-", "")]),
]


def _values_present(data: dict, val_key: str) -> bool:
    return any(v is not None for v in data[val_key])


def plot_task(task: str, runs: list[tuple[RunInfo, dict]], out_dir: pathlib.Path):
    """Create one figure per task with subplots for each available metric.

    Runs that share the same base_config_name (i.e. differ only by seed) are
    grouped: if the group has >1 seed, a mean line + shaded ±1 std region is
    plotted instead of individual lines.
    """

    # ── Group by (model_type, base_config_name) ──────────────────────────────
    GroupEntry = collections.namedtuple("GroupEntry", ["label", "data_list"])
    groups: dict[tuple, GroupEntry] = {}
    for run_info, data in runs:
        key = (run_info.model_type, run_info.base_config_name)
        if key not in groups:
            groups[key] = GroupEntry(label=run_info.label, data_list=[])
        groups[key].data_list.append(data)
    group_list = list(groups.values())

    # Decide which metrics are non-empty across all groups for this task
    active_metrics = []
    for title, log_scale, series_list in METRICS:
        for group in group_list:
            if any(
                _values_present(d, vk)
                for d in group.data_list
                for vk, _, _, _ in series_list
            ):
                active_metrics.append((title, log_scale, series_list))
                break

    if not active_metrics:
        print(f"  No data for task {task}, skipping.")
        return

    n_metrics = len(active_metrics)
    NCOLS = 3
    nrows = math.ceil(n_metrics / NCOLS)
    ncols = min(n_metrics, NCOLS)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )
    axes_flat = axes.flatten()
    # Hide any unused subplot cells
    for i in range(n_metrics, len(axes_flat)):
        axes_flat[i].set_visible(False)

    colors = cm.tab10(np.linspace(0, 1, max(len(group_list), 1)))

    for ax, (metric_name, log_scale, series_list) in zip(axes_flat, active_metrics):
        for group, color in zip(group_list, colors):
            for val_key, epoch_key, linestyle, suffix in series_list:
                # Collect non-None (x, y) pairs from every seed in this group
                all_series = []
                for data in group.data_list:
                    pairs = [
                        (x, y)
                        for x, y in zip(data[epoch_key], data[val_key])
                        if y is not None
                    ]
                    if pairs:
                        all_series.append(pairs)

                if not all_series:
                    continue

                line_label = group.label + suffix

                if len(all_series) == 1:
                    xs_clean, ys_clean = zip(*all_series[0])
                    ax.plot(
                        xs_clean, ys_clean,
                        label=line_label, color=color,
                        linestyle=linestyle, linewidth=1.5,
                    )
                else:
                    # Multiple seeds: union of all x-axes, per-epoch stats from
                    # however many seeds have data at each epoch.
                    all_xs = sorted(
                        set.union(*[{x for x, _ in s} for s in all_series])
                    )
                    if not all_xs:
                        continue
                    seed_dicts = [dict(s) for s in all_series]
                    means, stds, counts = [], [], []
                    for x in all_xs:
                        ys_at_x = [sd[x] for sd in seed_dicts if x in sd]
                        means.append(np.mean(ys_at_x))
                        stds.append(np.std(ys_at_x) if len(ys_at_x) > 1 else 0.0)
                        counts.append(len(ys_at_x))
                    means  = np.array(means)
                    stds   = np.array(stds)
                    counts = np.array(counts)
                    n_seeds = len(all_series)
                    ax.plot(
                        all_xs, means,
                        label=f"{line_label} (n={n_seeds} seeds)",
                        color=color, linestyle=linestyle, linewidth=1.5,
                    )
                    # Shade ±1 std only where >1 seed contributed
                    multi_mask = counts > 1
                    if multi_mask.any():
                        shaded_means = np.where(multi_mask, means, np.nan)
                        shaded_stds  = np.where(multi_mask, stds,  np.nan)
                        ax.fill_between(
                            all_xs,
                            shaded_means - shaded_stds,
                            shaded_means + shaded_stds,
                            color=color, alpha=0.2,
                        )

        ax.set_title(metric_name)
        ax.set_xlabel("Epoch")
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # Single shared legend below all subplots
    handles, labels = axes_flat[0].get_legend_handles_labels()
    # Collect from all axes in case some only appear in later subplots
    for ax in axes_flat[1:n_metrics]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(len(group_list), 3),
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
    print(f"Found {len(runs)} runs matching task-specific config filters.\n")

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
