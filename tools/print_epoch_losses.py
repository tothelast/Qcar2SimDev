#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable, List, Tuple
import wandb
from wandb.proto import wandb_internal_pb2 as wandb_pb2
from wandb.sdk.internal import datastore

def find_wandb_file(run_dir: Path) -> Path:
    wandb_dir = run_dir / "wandb"
    if not wandb_dir.exists():
        raise FileNotFoundError(f"No 'wandb' directory found under {run_dir}")
    event_files = list(wandb_dir.glob("offline-run-*/run-*.wandb"))
    if not event_files:
        raise FileNotFoundError(f"No run-*.wandb files found inside {wandb_dir}.")
    return max(event_files, key=lambda p: p.stat().st_mtime)

def iter_history_records(wandb_file: Path) -> Iterable[dict]:
    wandb._assert_is_internal_process = True
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
                continue
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
    series: DefaultDict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row in records:
        step = row.get("trainer/global_step") or row.get("_step")
        if step is None:
            continue
        epoch = row.get("epoch")
        
        for key, value in row.items():
            if not isinstance(value, (int, float)):
                continue
            # Store tuple of (step, value, epoch)
            # We use a special key for epoch to track it
            if key == "epoch":
                series["epoch"].append((float(step), float(value)))
            else:
                series[key].append((float(step), float(value)))
    
    for key, values in series.items():
        values.sort(key=lambda item: item[0])
    return series

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()

    wandb_file = find_wandb_file(args.run_dir)
    records = list(iter_history_records(wandb_file))
    
    # We want to group by epoch
    epoch_data = defaultdict(dict)
    
    for row in records:
        epoch = row.get("epoch")
        if epoch is None:
            continue
        
        # Round epoch to integer if it's close (wandb sometimes logs partial epochs)
        # But usually 'epoch' metric is logged as integer at end of epoch or continuous float
        # Let's look for 'train/loss_epoch' which is usually logged once per epoch
        
        if "train/loss_epoch" in row:
            epoch_idx = int(epoch)
            epoch_data[epoch_idx]["train_loss"] = row["train/loss_epoch"]
            
        if "val/loss" in row:
            epoch_idx = int(epoch)
            epoch_data[epoch_idx]["val_loss"] = row["val/loss"]
            
        if "val_losses/route_loss" in row:
            epoch_idx = int(epoch)
            epoch_data[epoch_idx]["val_route_loss"] = row["val_losses/route_loss"]
            
        if "val_losses/speed_wps_loss" in row:
            epoch_idx = int(epoch)
            epoch_data[epoch_idx]["val_speed_loss"] = row["val_losses/speed_wps_loss"]

    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Val Loss':<12} | {'Val Route':<12} | {'Val Speed':<12}")
    print("-" * 66)
    
    sorted_epochs = sorted(epoch_data.keys())
    for epoch in sorted_epochs:
        train_loss = epoch_data[epoch].get("train_loss", "N/A")
        val_loss = epoch_data[epoch].get("val_loss", "N/A")
        val_route = epoch_data[epoch].get("val_route_loss", "N/A")
        val_speed = epoch_data[epoch].get("val_speed_loss", "N/A")
        
        t_str = f"{train_loss:.6f}" if isinstance(train_loss, float) else str(train_loss)
        v_str = f"{val_loss:.6f}" if isinstance(val_loss, float) else str(val_loss)
        vr_str = f"{val_route:.6f}" if isinstance(val_route, float) else str(val_route)
        vs_str = f"{val_speed:.6f}" if isinstance(val_speed, float) else str(val_speed)
        
        print(f"{epoch:<6} | {t_str:<12} | {v_str:<12} | {vr_str:<12} | {vs_str:<12}")

if __name__ == "__main__":
    main()
