"""Run the pilot study and print its tables (run from VS Code with the Run button, or: python run_pilot.py)."""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

WORKERS = 12    # workers you will use for the real experiments (for example cores - 2)
HOURS = 100     # total hours you can spend on the experiments

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ["OMP_NUM_THREADS"] = "1"

cmd = [sys.executable, "-m", "mctsjssp", "pilot", "--config", "configs/pilot.yaml",
       "--set", f"pilot.workers={WORKERS}", "--set", f"pilot.available_hours={HOURS}"]
print("running:", " ".join(cmd))
subprocess.run(cmd, check=True)

out = ROOT / "results" / "pilot"
pd.set_option("display.width", 200, "display.max_columns", 30)
for name in ["timing", "by_configuration", "design", "training"]:
    f = out / f"{name}.csv"
    if f.exists():
        print(f"\n=== {name} ===")
        print(pd.read_csv(f).round(4).to_string(index=False))
print(f"\nCSV files saved in: {out}")
