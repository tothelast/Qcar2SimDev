import wandb
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as wandb_pb2
import json
import sys
from pathlib import Path

def inspect_wandb(wandb_path):
    print(f"Inspecting {wandb_path}")
    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_path))
    
    while True:
        data = ds.scan_data()
        if data is None:
            break
        
        record = wandb_pb2.Record()
        record.ParseFromString(data)
        
        if record.WhichOneof("record_type") == "run":
            run = record.run
            print("Run Config:")
            try:
                config = json.loads(run.config_json) if run.config_json else {}
                print(json.dumps(config, indent=2))
            except:
                print("Could not parse config_json")
            return # Found config, exit

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_wandb.py <path_to_wandb_file>")
        sys.exit(1)
    inspect_wandb(sys.argv[1])
