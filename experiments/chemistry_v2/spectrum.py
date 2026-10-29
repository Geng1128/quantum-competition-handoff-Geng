"""Rank lower bound for the two-body factorization.

``sum_(rho c) X (x) X`` is a PSD matrix of rank at most ``R*C`` on the
``N^2`` pair space, so the best rank-L PSD approximation of the shifted target
(top-L positive eigenpairs) bounds ``error_2`` from below for ``R*C = L``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import refdata


def spectrum(target: np.ndarray) -> np.ndarray:
    n2 = target.shape[0] * target.shape[1]
    matrix = target.reshape(n2, n2)
    matrix = 0.5 * (matrix + matrix.T)
    return np.linalg.eigvalsh(matrix)[::-1]


def min_rank_for(eigs: np.ndarray, tolerance: float) -> int:
    positive = np.clip(eigs, 0.0, None)
    negative_sq = float(np.sum(np.square(np.clip(eigs, None, 0.0))))
    tail_sq = np.cumsum(np.square(positive[::-1]))[::-1]  # tail_sq[L] = sum_{i>=L} pos^2
    tail_sq = np.append(tail_sq, 0.0)
    errors = np.sqrt(tail_sq + negative_sq)
    hits = np.nonzero(errors <= tolerance)[0]
    return int(hits[0]) if hits.size else -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=refdata.DEFAULT_CACHE.parent / "spectrum.json")
    args = parser.parse_args()
    ref = refdata.load()
    eigs = spectrum(ref.two_body_target())
    negative = eigs[eigs < 0]
    report = {
        "num_positive": int(np.sum(eigs > 0)),
        "negative_frobenius": float(np.sqrt(np.sum(negative**2))),
        "top_eigs": eigs[:10].tolist(),
        "min_rank": {f"{t:.3f}": min_rank_for(eigs, t) for t in (0.005, 0.01, 0.015, 0.018, 0.02)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_suffix(".npy"), eigs)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
