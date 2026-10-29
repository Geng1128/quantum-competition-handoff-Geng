"""Extract public chemistry factorizations and report shapes / metrics."""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

import refdata

if str(refdata.VERIFIERS_DIR) not in sys.path:
    sys.path.insert(0, str(refdata.VERIFIERS_DIR))
from dfthcsos.verify import native_inputs  # noqa: E402

PUBLIC = refdata.PROJECT_ROOT / "external" / "public_v2"
OUT = refdata.PROJECT_ROOT / "runs" / "chem_v2" / "public_factorizations"
SOURCES = {
    "mrmeows_r4": "mrmeows__qcc-13-75-open-76q-invsqrt-448q-chemistry/submission.zip",
    "busyaprime_r1": "busyaprime__33-lb-11-94676-78-qubit-invsqrt-all-verified/submission.zip",
    "rustad_r10": "michaelrustad__vqcc-r10-merged-group-chemistry-12-87-lb/submission.zip",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, rel in SOURCES.items():
        with zipfile.ZipFile(PUBLIC / rel) as archive:
            raw = archive.read("dfthcsos/submission.npz")
        (OUT / f"{name}.npz").write_bytes(raw)
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            params = {k: np.array(data[k]) for k in data.files}
        w, W, _, _ = native_inputs(params)
        report[name] = {
            "files": sorted(params),
            "U": list(params["U"].shape),
            "W": list(params["W"].shape),
            "tree_edges_head": params["tree_edges"][:6].tolist(),
            "w_abs_mean": float(np.mean(np.abs(w))),
            "W_offset_abs_mean": float(np.mean(np.abs(W))),
        }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
