"""Evaluate a trained factorization's angle words with the official metric code."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import refdata
from costmodel import shape_cost

if str(refdata.VERIFIERS_DIR) not in sys.path:
    sys.path.insert(0, str(refdata.VERIFIERS_DIR))
from dfthcsos.scoreinfra.dfthc_math import evaluate_dfthc_angle_words  # noqa: E402
from dfthcsos.verify import native_inputs  # noqa: E402


def evaluate(path: Path, J: int | None = None) -> dict:
    ref = refdata.load()
    with np.load(path, allow_pickle=False) as data:
        params = {k: np.array(data[k]) for k in data.files}
    w, W, _, _ = native_inputs(params)
    brot = int(params["brot"])
    metrics = evaluate_dfthc_angle_words(
        params["angle_words"].astype(np.int64), w, W, ref.S, ref.beta_1,
        {"oei": ref.oei, "tei": ref.tei, "electrons": ref.electrons, "ecore": ref.ecore},
        brot=brot, tree_edges=params["tree_edges"].astype(np.int64), energy_upper=refdata.UPPER_ENERGY,
    )
    R, B, C = w.shape
    passes = (
        metrics["error_1"] <= refdata.ERROR_THRESHOLD
        and metrics["error_2"] <= refdata.ERROR_THRESHOLD
        and metrics["gap_in_domain"]
        and metrics["E_gap"] > refdata.E_GAP_FLOOR
    )
    J = J or max(j for j in range(1, 12) if 11 + j * brot + 75 <= 128)
    cost = shape_cost(R, B, C, brot, J)
    return {
        "shape": [R, B, C],
        "brot": brot,
        "metrics": metrics,
        "passes": bool(passes),
        "model_T": cost.toffoli,
        "model_Q": cost.qubits,
        "model_score": cost.score(metrics["lambda_eff"]) if passes else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path, nargs="+")
    args = parser.parse_args()
    for p in args.npz:
        print(p.name, json.dumps(evaluate(p), default=float))


if __name__ == "__main__":
    main()
