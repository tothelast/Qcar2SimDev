import wandb
import sys
import os

def inspect_run(run_path):
    print(f"Inspecting run: {run_path}")
    try:
        # api = wandb.Api()

        from wandb.sdk.data_types.base_types.wb_value import WBValue
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal import datastore
        
        # Find the .wandb file
        wandb_file = None
        for root, dirs, files in os.walk(run_path):
            for file in files:
                if file.endswith(".wandb"):
                    wandb_file = os.path.join(root, file)
                    break
        
        if not wandb_file:
            print("No .wandb file found.")
            return

        print(f"Found .wandb file: {wandb_file}")
        
        ds = datastore.DataStore()
        ds.open_for_scan(wandb_file)
        
        keys = set()
        history_count = 0
        
        while True:
            data = ds.scan_record()
            if data is None:
                break
            
            # Debugging: print type and first few bytes
            # print(f"Data type: {type(data)}")
            # print(f"Data repr: {repr(data)}")

            blob = data
            if isinstance(data, tuple):
                blob = data[-1]
            
            try:
                pb = wandb_internal_pb2.Record()
                pb.ParseFromString(blob)
                
                if pb.HasField("history"):
                    history = pb.history
                    for item in history.item:
                        keys.add(item.key)
                        # Check if we have annotation counts
                        if "num_annotations" in item.key or "samples" in item.key:
                            print(f"Found key: {item.key} -> {item.value_json}")
                    history_count += 1
                elif pb.HasField("summary"):
                    summary = pb.summary
                    for item in summary.update:
                        keys.add(item.key)
                        if "num_annotations" in item.key or "samples" in item.key:
                            print(f"Found summary key: {item.key} -> {item.value_json}")
                            
            except Exception as e:
                # print(f"Parse error: {e}")
                pass
            
            if pb.HasField("history"):
                history = pb.history
                for item in history.item:
                    keys.add(item.key)
                history_count += 1
                
        print(f"Total history records: {history_count}")
        print(f"Available keys: {sorted(list(keys))}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_wandb.py <run_dir>")
    else:
        inspect_run(sys.argv[1])
