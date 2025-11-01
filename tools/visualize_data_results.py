import gzip

from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent  # repo root
results_path = base_dir / "database/data/simlingo/routes_training/qlabs/Rep_full_circuit_20251027_173348/TownQLabs/measurements/0000.json.gz"
with gzip.open(results_path, "rt") as f:
    print(f.read())