import json
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from omegaconf import OmegaConf
from transformers import AutoProcessor

from simlingo_training.config import TrainConfig
from simlingo_training.models.driving import DrivingModel, decode_uint8
from simlingo_training.utils.custom_types import DrivingExample


def move_to_device(obj, device):
    """Recursively move tensors in nested structures to the given device."""
    import torch as _torch

    if isinstance(obj, _torch.Tensor):
        return obj.to(device)
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):  # namedtuple
        return type(obj)(*(move_to_device(x, device) for x in obj))
    if isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    return obj


def extract_run_id_from_measurement_path(measurement_path: str) -> str:
    """Return a per-run identifier from a measurement file path.

    Expected pattern (QLabs):
    .../routes_validation/qlabs/Rep_<route>_<timestamp>/TownQLabs/measurements/0000.json.gz
    We use the parent of the measurements folder (e.g. .../Rep_.../TownQLabs) as run id.
    """
    p = Path(measurement_path)
    # p ... /measurements/0000.json.gz -> parent is measurements, parent.parent is TownQLabs
    run_dir = p.parent.parent
    return str(run_dir)


def compute_ade_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Average displacement error (ADE) per sample.

    pred, gt: [B, F, 2] = (x, y) ego-frame waypoints for the same F future steps.
    For each sample b: ADE_b = (1/F) * sum_f ||pred[b, f] - gt[b, f]||_2 (in metres).
    Returns: [B].
    """
    diff = pred - gt
    dists = torch.linalg.norm(diff, dim=-1)
    return dists.mean(dim=-1)


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    """Offline ADE-based evaluation on QLabs validation runs.

    Usage (example):
      python simlingo/simlingo_training/eval_qlabs_offline_ade.py \
        checkpoint=/path/to/outputs/.../checkpoints/epoch=013.ckpt
    """
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed, workers=True)

    if cfg.checkpoint is None:
        raise ValueError("Please provide checkpoint=/path/to/checkpoint.ckpt")

    load_path = Path(cfg.checkpoint)
    if not load_path.is_file() and not load_path.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    # Load original training config from the checkpoint's Hydra config
    config_path = load_path.parent.parent / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Hydra config not found next to checkpoint: {config_path}")

    cfg_loaded = OmegaConf.load(config_path)
    cfg_loaded.checkpoint = str(load_path)
    cfg = cfg_loaded  # type: ignore

    # Ensure we evaluate driving only, without augmentation or QA/commentary
    cfg.data_module.dreamer_dataset = None
    cfg.data_module.qa_dataset = None
    cfg.data_module.insteval_dataset = None
    cfg.data_module.base_dataset.img_augmentation = False
    cfg.data_module.base_dataset.img_shift_augmentation = False
    cfg.data_module.base_dataset.use_commentary = False
    cfg.data_module.base_dataset.use_qa = False

    processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)
    cache_dir = None

    data_module = hydra.utils.instantiate(
        cfg.data_module,
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        _recursive_=False,
    )
    data_module.setup()

    model: DrivingModel = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=cache_dir,
        _recursive_=False,
    )

    # Support both standard Lightning checkpoints (single file) and
    # DeepSpeed ZeRO checkpoints stored in a directory.
    if load_path.is_dir():
        state_dict = get_fp32_state_dict_from_zero_checkpoint(str(load_path))
    else:
        state_dict = torch.load(str(load_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    val_loader = data_module.val_dataloader()

    per_run_ade_model_vs_teleop: Dict[str, List[float]] = {}
    per_run_ade_model_vs_route: Dict[str, List[float]] = {}
    per_run_ade_teleop_vs_route: Dict[str, List[float]] = {}

    with torch.no_grad():
        for batch in val_loader:
            batch: DrivingExample = move_to_device(batch, device)
            run_ids = decode_uint8(batch.run_id)

            speed_wps, _, _ = model.forward(batch, return_language=False)
            # speed_wps: [B, F, 2]
            gt_wps = batch.driving_label.waypoints  # [B, F, 2]

            # Align route polyline to the same future horizon F
            route_path = batch.driving_label.path  # [B, R, 2]
            if route_path.shape[1] < speed_wps.shape[1]:
                raise ValueError(
                    f"Route path has fewer points ({route_path.shape[1]}) than future waypoints ({speed_wps.shape[1]})."
                )
            route_path = route_path[:, : speed_wps.shape[1], :]

            ade_model_vs_teleop = compute_ade_per_sample(speed_wps, gt_wps).cpu().numpy()
            ade_model_vs_route = compute_ade_per_sample(speed_wps, route_path).cpu().numpy()
            ade_teleop_vs_route = compute_ade_per_sample(gt_wps, route_path).cpu().numpy()

            for ade_mt, ade_mr, ade_tr, measurement_path in zip(
                ade_model_vs_teleop, ade_model_vs_route, ade_teleop_vs_route, run_ids
            ):
                run_id = extract_run_id_from_measurement_path(measurement_path)
                per_run_ade_model_vs_teleop.setdefault(run_id, []).append(float(ade_mt))
                per_run_ade_model_vs_route.setdefault(run_id, []).append(float(ade_mr))
                per_run_ade_teleop_vs_route.setdefault(run_id, []).append(float(ade_tr))

    if not per_run_ade_model_vs_teleop:
        raise RuntimeError("No ADE values computed; check that validation data is configured correctly.")

    # Aggregate per run
    ade_model_vs_teleop_per_run = {
        run_id: float(np.mean(values)) for run_id, values in per_run_ade_model_vs_teleop.items()
    }
    ade_model_vs_route_per_run = {
        run_id: float(np.mean(values)) for run_id, values in per_run_ade_model_vs_route.items()
    }
    ade_teleop_vs_route_per_run = {
        run_id: float(np.mean(values)) for run_id, values in per_run_ade_teleop_vs_route.items()
    }

    tau = 2.0  # metres, scale for converting ADE to a reward in [0, 1] (model vs teleop only)
    reward_per_run = {run_id: float(np.exp(-ade / tau)) for run_id, ade in ade_model_vs_teleop_per_run.items()}

    mean_ade_model_vs_teleop = float(np.mean(list(ade_model_vs_teleop_per_run.values())))
    mean_ade_model_vs_route = float(np.mean(list(ade_model_vs_route_per_run.values())))
    mean_ade_teleop_vs_route = float(np.mean(list(ade_teleop_vs_route_per_run.values())))
    mean_reward = float(np.mean(list(reward_per_run.values())))

    results = {
        "tau": tau,
        "num_runs": len(ade_model_vs_teleop_per_run),
        # Backwards-compatible: keep mean_ADE/per_run_ADE as model-vs-teleop metrics
        "mean_ADE": mean_ade_model_vs_teleop,
        "mean_reward": mean_reward,
        "per_run_ADE": ade_model_vs_teleop_per_run,
        "per_run_reward": reward_per_run,
        # Explicit metric names for route-based analysis
        "mean_ADE_model_vs_teleop": mean_ade_model_vs_teleop,
        "mean_ADE_model_vs_route": mean_ade_model_vs_route,
        "mean_ADE_teleop_vs_route": mean_ade_teleop_vs_route,
        "per_run_ADE_model_vs_teleop": ade_model_vs_teleop_per_run,
        "per_run_ADE_model_vs_route": ade_model_vs_route_per_run,
        "per_run_ADE_teleop_vs_route": ade_teleop_vs_route_per_run,
    }

    run_dir = load_path.parent.parent
    out_dir = run_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "offline_ade_qlabs.json"

    with out_file.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved offline ADE results to {out_file}")


if __name__ == "__main__":
    main()

