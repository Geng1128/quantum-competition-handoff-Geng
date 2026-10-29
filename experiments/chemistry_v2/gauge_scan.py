"""Compare Givens-tree and sign gauges for nearest 8-bit rounding.

This is a no-training screen: the continuous vectors and weights stay fixed,
so a lower ``error_2`` directly identifies a representation with less
rounding noise before progressive quantization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import dfthc_torch as dt
import givens
import refdata
from gtrain import projector_design, refit_block, residual, xmat


def valid_tree(rng: np.random.Generator, n: int) -> np.ndarray:
    """Create an ordered rooted tree accepted by the Givens encoder."""
    edges = []
    for child in range(1, n):
        parent = int(rng.integers(0, child))
        edges.append((parent, child))
    return np.asarray(edges, dtype=np.int64)


def eval_words(problem, U: np.ndarray, w: torch.Tensor, edges: np.ndarray, brot: int) -> float:
    words = givens.to_words(givens.encode_numpy(U.reshape(-1, U.shape[-1]), edges), brot)
    uq = givens.decode_numpy(givens.words_to_theta(words, brot), edges).reshape(U.shape)
    uq_t = torch.tensor(uq, dtype=torch.float32, device=problem.device)
    A = projector_design(uq_t)
    wc = w.clone()
    with torch.no_grad():
        for r in range(wc.shape[0]):
            wc[r] = refit_block(problem, A, wc, r, wc.shape[2])
    return float(residual(problem, xmat(A, wc)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--brot", type=int, default=8)
    parser.add_argument("--trees", type=int, default=8)
    parser.add_argument("--gauges", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    problem = dt.Problem.build(refdata.load())
    with np.load(args.start, allow_pickle=False) as data:
        theta = np.asarray(data["theta"])
        w = torch.tensor(np.asarray(data["w"]), dtype=torch.float32, device=problem.device)
        base_edges = np.asarray(data["tree_edges"], dtype=np.int64)
    R, B, N1 = w.shape[0], w.shape[1], theta.shape[1] + 1
    U = givens.decode_numpy(theta, base_edges).reshape(R, B, N1)

    rows = []
    rows.append({"kind": "base", "tree": "source", "gauge": "none",
                 "error_2": eval_words(problem, U, w, base_edges, args.brot)})
    for i in range(args.gauges):
        signs = rng.choice(np.array([-1.0, 1.0]), size=(R, B, 1))
        rows.append({"kind": "sign", "tree": "source", "gauge": i,
                     "error_2": eval_words(problem, U * signs, w, base_edges, args.brot)})
    for i in range(args.trees):
        edges = valid_tree(rng, N1)
        rows.append({"kind": "tree", "tree": i, "gauge": "none",
                     "error_2": eval_words(problem, U, w, edges, args.brot),
                     "tree_edges": edges.tolist()})

    rows.sort(key=lambda row: row["error_2"])
    result = {"start": str(args.start), "shape": [R, B, int(w.shape[2])],
              "brot": args.brot, "seed": args.seed, "rows": rows,
              "best": rows[0], "base": next(row for row in rows if row["kind"] == "base")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({"base": result["base"], "best": result["best"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
