import os
import sys
import json
import glob
import torch
import numpy as np
import hydra
from pathlib import Path
from omegaconf import OmegaConf
from transformers import AutoProcessor, AutoTokenizer
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from tqdm import tqdm

# Add simlingo to python path
sys.path.append(os.path.join(os.getcwd(), 'simlingo'))

from simlingo_training.utils.custom_types import DrivingExample

def calculate_ade(route_preds, route_gt):
    return np.mean(np.linalg.norm(route_preds - route_gt, axis=-1))

def to_device(obj, device, dtype=None):
    if torch.is_tensor(obj):
        if dtype is not None and obj.is_floating_point():
            return obj.to(device=device, dtype=dtype)
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device, dtype) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_device(v, device, dtype) for v in obj]
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'): # NamedTuple
        return type(obj)(*[to_device(v, device, dtype) for v in obj])
    else:
        return obj

@hydra.main(config_path=".", config_name="my_config", version_base="1.1")
def main(cfg):
    print("Starting ADE calculation...")
    
    # Setup paths
    checkpoints_dir = "/home/garegin/Documents/Projects/Qcar2SimDev/simlingo/outputs/2025_11_26_18_06_21_qlabs_roundabout_finetune/checkpoints"
    validation_data_path = "/home/garegin/Documents/Projects/Qcar2SimDev/database/data/simlingo/routes_validation/qlabs"
    
    # Override config for validation
    # cfg.data_module.base_dataset.data_path = "database"
    # cfg.data_module.driving_dataset.data_path = "database"
    
    # Setup model and data module
    if "2B" in cfg.model.language_model.variant:
        processor = AutoTokenizer.from_pretrained(cfg.model.language_model.variant, trust_remote_code=True, use_fast=False)
    else:
        processor = AutoProcessor.from_pretrained(cfg.model.language_model.variant, trust_remote_code=True, use_fast=False)
        
    model_type_name = cfg.model.vision_model.variant.split('/')[1]
    cache_dir = f"pretrained/{(model_type_name)}"
    
    # Initialize DataModule
    # We need to manually set the validation dataset path or ensure the config points to it
    # The datamodule uses base_dataset.data_path combined with other params.
    # We'll rely on the config override above.
    
    data_module = hydra.utils.instantiate(
        cfg.data_module, 
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        predict=False,
        _recursive_=False
    )
    data_module.setup()
    val_loader = data_module.val_dataloader()
    
    # Initialize Model
    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=cache_dir,
        _recursive_=False
    )
    model.eval()
    model.half()
    model.cuda()
    
    results = {}
    
    # Iterate over checkpoints
    checkpoint_dirs = sorted(glob.glob(os.path.join(checkpoints_dir, "epoch=*.ckpt")))
    
    for ckpt_path in checkpoint_dirs:
        epoch_name = os.path.basename(ckpt_path).split('.ckpt')[0]
        print(f"Processing {epoch_name}...")
        
        try:
            # Load checkpoint
            if os.path.isdir(ckpt_path):
                state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_path)
            else:
                state_dict = torch.load(ckpt_path, map_location="cpu")
            
            model.load_state_dict(state_dict)
            del state_dict
            torch.cuda.empty_cache()
            
            all_ade = []
            
            print(f"Validation batches: {len(val_loader)}")
            
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_loader, desc=f"Evaluating {epoch_name}")):
                    if i % 50 != 0:
                        continue
                        
                    # Move batch to device
                    batch = to_device(batch, model.device, dtype=torch.float16)
                    
                    speed_wps, route, language = model(batch, return_language=True)
                    
                    # Get GT
                    route_gt = batch.driving_label.path.cuda()
                    
                    # Calculate ADE for this batch
                    # route: [B, N, 2], route_gt: [B, N, 2]
                    # Ensure they are on same device and numpy
                    route_np = route.cpu().numpy()
                    route_gt_np = route_gt.cpu().numpy()
                    
                    batch_ade = calculate_ade(route_np, route_gt_np)
                    all_ade.append(batch_ade)
            
            epoch_ade = np.mean(all_ade)
            results[epoch_name] = float(epoch_ade)
            print(f"{epoch_name} ADE: {epoch_ade}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing {epoch_name}: {e}")
            
    # Save results
    with open("ade_results_all_epochs.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Results saved to ade_results_all_epochs.json")

if __name__ == "__main__":
    main()
