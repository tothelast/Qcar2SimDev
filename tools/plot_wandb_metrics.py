#!/usr/bin/env python3
"""
Script to plot training and validation metrics from wandb offline data.

Usage:
    python tools/plot_wandb_metrics.py <wandb_dir_or_file> [--output plot.png]

Examples:
    python tools/plot_wandb_metrics.py simlingo/outputs/2025_11_26_18_06_21_qlabs_roundabout_finetune/wandb/latest-run
    python tools/plot_wandb_metrics.py simlingo/outputs/2025_11_26_18_06_21_qlabs_roundabout_finetune --output metrics.png
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as wandb_pb2


def find_wandb_file(path: str) -> str:
    """Find the .wandb file from a path (file or directory)."""
    p = Path(path)

    if p.is_file() and p.suffix == '.wandb':
        return str(p)

    if p.is_dir():
        # Check for latest-run symlink
        latest_run = p / 'latest-run'
        if latest_run.exists():
            p = latest_run

        # Check for wandb subdir
        wandb_dir = p / 'wandb'
        if wandb_dir.exists():
            latest_run = wandb_dir / 'latest-run'
            if latest_run.exists():
                p = latest_run

        # Find .wandb file
        wandb_files = list(p.glob('*.wandb'))
        if not wandb_files:
            wandb_files = list(p.glob('**/*.wandb'))

        if wandb_files:
            return str(wandb_files[0])

    raise FileNotFoundError(f"No .wandb file found in {path}")


def load_wandb_metrics(wandb_path: str) -> Dict[str, Dict[str, List]]:
    """
    Load metrics from a wandb file.
    
    Returns a dict with metric names as keys and dict with 'step' and 'value' lists.
    """
    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_path))
    
    metrics = defaultdict(lambda: {'step': [], 'value': []})
    
    while True:
        data = ds.scan_data()
        if data is None:
            break
        
        record = wandb_pb2.Record()
        record.ParseFromString(data)
        record_type = record.WhichOneof('record_type')
        
        if record_type == 'history':
            hist = record.history
            step = hist.step.num if hist.step.num > 0 else None

            # Parse history items
            row = {}
            for item in hist.item:
                if item.value_json:
                    try:
                        val = json.loads(item.value_json)
                        # Key can be in item.key or item.nested_key
                        key = item.key if item.key else '/'.join(item.nested_key)
                        if key:
                            row[key] = val
                    except json.JSONDecodeError:
                        pass

            # Store metrics we care about
            step_val = step if step is not None else row.get('trainer/global_step', 0)
            if step_val is not None:
                for key, value in row.items():
                    if isinstance(value, (int, float)) and not key.startswith('_'):
                        metrics[key]['step'].append(step_val)
                        metrics[key]['value'].append(value)
    
    return dict(metrics)


def smooth(values: np.ndarray, weight: float = 0.9) -> np.ndarray:
    """Apply exponential moving average smoothing."""
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i-1] + (1 - weight) * values[i]
    return smoothed


def plot_metrics(metrics: Dict, output_path: Optional[str] = None, smooth_weight: float = 0.9):
    """Plot training and validation metrics."""

    # Define the metrics to plot
    metric_groups = {
        'Total Loss': {
            'train': ['train_losses/loss', 'train/loss_step'],
            'val': ['val_losses/loss', 'val/loss'],
        },
        'Route Loss': {
            'train': ['train_losses/route_loss'],
            'val': ['val_losses/route_loss'],
        },
        'Speed WPs Loss': {
            'train': ['train_losses/speed_wps_loss'],
            'val': ['val_losses/speed_wps_loss'],
        },
        'Language Loss': {
            'train': ['train_losses/language_loss'],
            'val': ['val_losses/language_loss'],
        },
        'Learning Rate': {
            'lr': ['lr-AdamW'],
        },
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    colors = {'train': 'blue', 'val': 'orange', 'lr': 'green'}

    for idx, (title, group) in enumerate(metric_groups.items()):
        ax = axes[idx]

        for mode, metric_names in group.items():
            for metric_name in metric_names:
                if metric_name in metrics and len(metrics[metric_name]['step']) > 0:
                    steps = np.array(metrics[metric_name]['step'])
                    values = np.array(metrics[metric_name]['value'])

                    # Sort by step
                    sort_idx = np.argsort(steps)
                    steps = steps[sort_idx]
                    values = values[sort_idx]

                    # Plot raw (faint) and smoothed
                    ax.plot(steps, values, alpha=0.2, color=colors[mode])
                    if len(values) > 1 and mode != 'lr':
                        smoothed = smooth(values, smooth_weight)
                        ax.plot(steps, smoothed, label=f'{mode} (smoothed)',
                               color=colors[mode], linewidth=2)
                    elif mode == 'lr':
                        ax.plot(steps, values, label='learning rate',
                               color=colors[mode], linewidth=2)
                    else:
                        ax.scatter(steps, values, label=mode, color=colors[mode], s=50)
                    break  # Only plot first found metric

        ax.set_xlabel('Global Step')
        ax.set_ylabel('Loss' if 'Loss' in title else title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Use scientific notation for learning rate
        if title == 'Learning Rate':
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    # Use last subplot for epoch markers if available
    ax = axes[5]
    if 'epoch' in metrics and len(metrics['epoch']['step']) > 0:
        steps = np.array(metrics['epoch']['step'])
        epochs = np.array(metrics['epoch']['value'])
        sort_idx = np.argsort(steps)
        ax.plot(steps[sort_idx], epochs[sort_idx], 'k-', linewidth=2)
        ax.set_xlabel('Global Step')
        ax.set_ylabel('Epoch')
        ax.set_title('Training Progress')
        ax.grid(True, alpha=0.3)
    else:
        ax.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from wandb data')
    parser.add_argument('wandb_path', type=str,
                       help='Path to wandb .wandb file or directory containing wandb data')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for the plot (default: show interactively)')
    parser.add_argument('--smooth', '-s', type=float, default=0.9,
                       help='Smoothing weight (0-1, higher = smoother)')
    args = parser.parse_args()

    # Find the wandb file
    wandb_file = find_wandb_file(args.wandb_path)
    print(f"Loading metrics from: {wandb_file}")

    metrics = load_wandb_metrics(wandb_file)

    print(f"Found {len(metrics)} metrics:")
    for name in sorted(metrics.keys()):
        print(f"  {name}: {len(metrics[name]['step'])} data points")

    plot_metrics(metrics, args.output, args.smooth)


if __name__ == '__main__':
    main()

