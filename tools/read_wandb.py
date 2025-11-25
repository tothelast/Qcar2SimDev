import wandb
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2
import sys
import json

def read_wandb_file(file_path):
    ds = DataStore()
    try:
        ds.open_for_scan(file_path)
    except Exception as e:
        print(f"Error opening file: {e}")
        return

    history = []
    summary = {}

    while True:
        data = ds.scan_record()
        if data is None:
            break
        
        if isinstance(data, tuple):
            # print(f"Data is tuple of length {len(data)}")
            data = data[1] # Assuming (offset, record)
        
        # print(f"Data type: {type(data)}, length: {len(data)}")
        try:
            pb = wandb_internal_pb2.Record()
            pb.ParseFromString(data)
        except Exception as e:
            print(f"Error parsing record: {e}")
            continue
        
        if pb.HasField('history'):
            item = {}
            for item_pb in pb.history.item:
                item[item_pb.key] = item_pb.value_json
            history.append(item)
        
        if pb.HasField('summary'):
            for item_pb in pb.summary.update:
                summary[item_pb.key] = item_pb.value_json

    print(f"Found {len(history)} history records.")
    if history:
        print("Last History Record:")
        print(json.dumps(history[-1], indent=2))
    
    print("\nSummary:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_wandb.py <path_to_wandb_file>")
        sys.exit(1)
    
    read_wandb_file(sys.argv[1])
