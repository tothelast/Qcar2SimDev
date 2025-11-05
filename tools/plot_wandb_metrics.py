#!/usr/bin/env python3
"""
Utility to extract fine-tuning metrics from an offline W&B run and plot them.

Example:
    python tools/plot_wandb_metrics.py \
        --run-dir simlingo/outputs/2025_10_28_00_46_56_qlabs_finetune \
        --output simlingo/outputs/2025_10_28_00_46_56_qlabs_finetune/metrics.png

Metric definitions (matching Simlingo fine-tuning):
    - Route loss: mean-squared error between the predicted future route waypoints
      (20 points in ego-frame) and the ground-truth route from measurements.
    - Speed/Δwaypoint loss: MSE over the speed target and waypoint deltas the model
      predicts for throttle/steering control (same targets the training pipeline
      uses during imitation learning).
    - Total loss: weighted sum of the component losses; this is what the optimiser minimizes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import wandb
from wandb.proto import wandb_internal_pb2 as wandb_pb2
from wandb.sdk.internal import datastore


def find_wandb_file(run_dir: Path) -> Path:
    """Return the newest W&B event file inside `run_dir`."""
    wandb_dir = run_dir / "wandb"
    if not wandb_dir.exists():
        raise FileNotFoundError(f"No 'wandb' directory found under {run_dir}")

    event_files = list(wandb_dir.glob("offline-run-*/run-*.wandb"))
    if not event_files:
        raise FileNotFoundError(
            f"No run-*.wandb files found inside {wandb_dir}. "
            "Did you run the fine-tune with WANDB_MODE=offline?"
        )

    return max(event_files, key=lambda p: p.stat().st_mtime)


def iter_history_records(wandb_file: Path) -> Iterable[dict]:
    """Yield decoded history records from a W&B offline event file."""
    wandb._assert_is_internal_process = True  # unlock datastore reader

    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_file))

    record = wandb_pb2.Record()
    while True:
        payload = ds.scan_data()
        if payload is None:
            break
        record.ParseFromString(payload)
        if record.WhichOneof("record_type") != "history":
            continue

        row = {}
        for item in record.history.item:
            if item.value_json == "":
                continue

            try:
                value = json.loads(item.value_json)
            except json.JSONDecodeError:
                # some image artifacts are not strict JSON; skip them
                continue

            # W&B stores hierarchical keys as nested_key entries.
            if item.key:
                key = item.key
            elif item.nested_key:
                key = "/".join(item.nested_key)
            else:
                continue

            row[key] = value

        if row:
            yield row


def collect_metric_series(records: Iterable[dict]) -> DefaultDict[str, List[Tuple[float, float]]]:
    """Convert iterable of history records into per-metric time series."""
    series: DefaultDict[str, List[Tuple[float, float]]] = defaultdict(list)

    for row in records:
        step = row.get("trainer/global_step")
        if step is None:
            step = row.get("_step")
        if step is None:
            continue

        for key, value in row.items():
            if key.startswith("_") or key.startswith("trainer/"):
                continue
            if not isinstance(value, (int, float)):
                continue

            series[key].append((float(step), float(value)))

    # Sort each metric by step to keep lines monotonic and easier to plot.
    for key, values in series.items():
        values.sort(key=lambda item: item[0])

    return series


def smooth_series(
    values: List[Tuple[float, float]], window: int
) -> List[Tuple[float, float]]:
    """Simple moving average smoothing for noisy per-step metrics."""
    if window <= 1 or len(values) <= window:
        return values
    steps, data = np.asarray(values).T
    kernel = np.ones(window, dtype=float) / window
    smoothed = np.convolve(data, kernel, mode="valid")
    smoothed_steps = steps[window - 1 :]
    return list(zip(smoothed_steps, smoothed))


def select_metrics(
    series: DefaultDict[str, List[Tuple[float, float]]],
    smooth_window: int = 200,
) -> Tuple[dict, dict]:
    """
    Keep only the most informative metrics and drop duplicates/irrelevant ones.

    Returns:
        (train_series, val_series) dictionaries with human-friendly labels.
    """
    desired_train = {
        "train/loss_epoch": "Train loss (epoch avg)",
        "train_losses/route_loss": "Train route error (smoothed)",
        "train_losses/speed_wps_loss": "Train speed/Δwp error (smoothed)",
    }
    desired_val = {
        "val/loss": "Val loss",
        "val_losses/route_loss": "Val route error",
        "val_losses/speed_wps_loss": "Val speed/Δwp error",
    }

    train_series, val_series = {}, {}

    for key, label in desired_train.items():
        if key not in series:
            continue
        values = series[key]
        if "smoothed" in label.lower():
            values = smooth_series(values, smooth_window)
        train_series[label] = values

    for key, label in desired_val.items():
        if key not in series:
            continue
        val_series[label] = series[key]

    if not val_series:
        raise RuntimeError("Validation metrics not found in W&B history.")

    return train_series, val_series


def plot_metrics(
    train_series: dict,
    val_series: dict,
    output_path: Path,
    show: bool = False,
) -> None:
    """Plot selected training/validation metrics and save to `output_path`."""
    # Two stacked plots: training on top, validation bottom.
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    train_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    val_colors = ["#1f77b4", "#9467bd", "#8c564b"]

    for (label, values), color in zip(train_series.items(), train_colors):
        steps, vals = np.asarray(values).T
        axes[0].plot(
            steps,
            vals,
            label=label,
            color=color,
            linewidth=2.0,
        )
    axes[0].set_title("Training (smoothed where appropriate)")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")

    for (label, values), color in zip(val_series.items(), val_colors):
        steps, vals = np.asarray(values).T
        axes[1].plot(
            steps,
            vals,
            label=label,
            color=color,
            linewidth=2.0,
        )
    axes[1].set_title("Validation")
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Global step")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def save_metric_description(run_dir: Path, smooth_window: int) -> None:
    """Output a companion figure explaining how each metric is computed."""
    explanation_lines = [
        "Metric definitions:",
        "• Route error – mean squared error between the model's 20 future ego-frame waypoints and the ground-truth route.",
        "• Speed/Δwaypoint error – mean squared error over the predicted target speed and waypoint deltas used for imitation learning.",
        "• Total loss – weighted sum of the component errors (same objective the optimiser minimises).",
        f"Training curves labelled 'smoothed' apply a {smooth_window}-step moving average to reduce per-frame noise.",
    ]
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.axis("off")
    ax.text(
        0.01,
        0.9,
        "\n".join(explanation_lines),
        ha="left",
        va="top",
        fontsize=11,
        wrap=True,
    )
    fig.tight_layout()
    output_path = run_dir / "metrics_explained.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot fine-tune metrics from an offline W&B run.")
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Path to the fine-tune output directory (the folder that contains checkpoints/ and wandb/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to save the plot PNG (default: <run_dir>/metrics.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving it.",
    )
    args = parser.parse_args()

    wandb_file = find_wandb_file(args.run_dir)
    records = list(iter_history_records(wandb_file))
    series = collect_metric_series(records)

    smooth_window = 200
    train_series, val_series = select_metrics(series, smooth_window=smooth_window)
    output_path = args.output or (args.run_dir / "metrics.png")
    plot_metrics(train_series, val_series, output_path, show=args.show)
    save_metric_description(args.run_dir, smooth_window)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
