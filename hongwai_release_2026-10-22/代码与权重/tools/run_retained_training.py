#!/usr/bin/env python3
"""Relocate a historical training configuration. Defaults to printing the command."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--config", required=True, type=Path)
p.add_argument("--dataset-dir", type=Path, default=ROOT/"datasets")
p.add_argument("--output-dir", type=Path)
p.add_argument("--device", default="cuda:0")
p.add_argument("--execute", action="store_true")
a = p.parse_args()
c = json.loads(a.config.read_text(encoding="utf-8"))
params = c["args"]
params["dataset_dir"] = str(a.dataset_dir.resolve())
params["device"] = a.device
params["output_dir"] = str((a.output_dir or ROOT/"results"/(a.config.stem+"_retrain")).resolve())
cmd = [sys.executable, str(ROOT/c["script"])]
for key, value in params.items():
    if isinstance(value, bool):
        if value: cmd.append("--"+key)
    else:
        for v in value if isinstance(value, list) else [value]:
            cmd += ["--"+key, str(v)]
print(shlex.join(cmd), flush=True)
if a.execute:
    subprocess.run(cmd, cwd=ROOT, check=True)
