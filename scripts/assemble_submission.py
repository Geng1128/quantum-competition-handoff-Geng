"""Assemble a submission.zip from per-task sources with pinned SHA-256 checks.

Usage:
  python scripts/assemble_submission.py --manifest manifest.json --out runs/x/submission.zip

The manifest maps each archive member to a source: either a file path or
``{"zip": path, "member": name}``, plus the expected ``sha256``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def read_source(spec: dict) -> bytes:
    if "zip" in spec:
        with zipfile.ZipFile(spec["zip"]) as archive:
            return archive.read(spec["member"])
    return Path(spec["path"]).read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    members = {}
    for name, spec in manifest["members"].items():
        data = read_source(spec)
        digest = hashlib.sha256(data).hexdigest()
        if digest != spec["sha256"]:
            raise SystemExit(f"hash mismatch for {name}: {digest} != {spec['sha256']}")
        members[name] = data
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(members):
            archive.writestr(name, members[name])
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(json.dumps({"out": str(args.out), "sha256": digest, "members": len(members)}))


if __name__ == "__main__":
    main()
