#!/usr/bin/env python3
"""Verify every file listed in the release manifest without ML dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--manifest", type=Path, default=root / "MANIFEST.json")
args = parser.parse_args()
manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
errors = []
for item in manifest["files"]:
    path = root / item["path"]
    if not path.is_file():
        errors.append(f"Missing: {item['path']}"); continue
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    if path.stat().st_size != item["bytes"] or digest.hexdigest() != item["sha256"]:
        errors.append(f"Mismatch: {item['path']}")
if errors:
    raise SystemExit("\n".join(errors))
print(f"PASS: {len(manifest['files'])} files verified")
