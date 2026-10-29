"""Sample exact internal Givens gauges and keep the locally best rounding.

Global vector sign is only one gauge.  Internal tree signs give 2^(N-1)
equivalent angle representations; this script samples that space and scores
each representation by the projector rounding error before evaluating the full
DF-THC residual.
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


def sample_gauge(theta: np.ndarray, edges: np.ndarray, rng: np.random.Generator, final_sign: int) -> np.ndarray:
    n_edges = len(edges)
    children = [[] for _ in range(n_edges + 1)]
    for t, (p, c) in enumerate(edges):
        children[int(p)].append(t)
    last = {p: rows[-1] for p, rows in enumerate(children) if rows}
    g = np.ones(n_edges + 1, dtype=np.int8)
    out = np.empty_like(theta)
    for t, (p, c) in enumerate(edges):
        p = int(p)
        c = int(c)
        parent_out = final_sign if last[p] == t else int(rng.choice((-1, 1)))
        child_out = int(rng.choice((-1, 1))) if children[c] else final_sign
        out[t] = np.arctan2(child_out * g[p] * np.sin(np.pi * theta[t]),
                            parent_out * g[p] * np.cos(np.pi * theta[t])) / np.pi
        g[p] = parent_out
        g[c] = child_out
    return out


def evaluate(problem, theta: np.ndarray, w: torch.Tensor, edges: np.ndarray, brot: int) -> float:
    words = givens.to_words(theta, brot)
    uq = givens.decode_numpy(givens.words_to_theta(words, brot), edges)
    uq_t = torch.tensor(uq.reshape(w.shape[0], w.shape[1], -1), dtype=torch.float32, device=problem.device)
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
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--brot", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    problem = dt.Problem.build(refdata.load())
    with np.load(args.start, allow_pickle=False) as data:
        theta = np.asarray(data["theta"], dtype=np.float64)
        w = torch.tensor(np.asarray(data["w"]), dtype=torch.float32, device=problem.device)
        edges = np.asarray(data["tree_edges"], dtype=np.int64)
    base_words = givens.to_words(theta, args.brot)
    base_theta = givens.words_to_theta(base_words, args.brot)
    base_u = givens.decode_numpy(theta, edges)
    best_words = base_words.copy()
    best_local = np.full(theta.shape[0], np.inf)
    for row in range(theta.shape[0]):
        u = base_u[row]
        p = np.outer(u, u)
        for sample in range(args.samples):
            gauged = sample_gauge(theta[row], edges, rng, 1 if sample % 2 else -1)
            words = givens.to_words(gauged[None], args.brot)[0]
            uq = givens.decode_numpy(givens.words_to_theta(words[None], args.brot), edges)[0]
            local = float(np.sum((np.outer(uq, uq) - p) ** 2))
            if local < best_local[row]:
                best_local[row] = local
                best_words[row] = words
    best_theta = givens.words_to_theta(best_words, args.brot)
    base_err = evaluate(problem, base_theta, w, edges, args.brot)
    best_err = evaluate(problem, best_theta, w, edges, args.brot)
    print(f"base nearest error_2 {base_err:.9f}", flush=True)
    print(f"sampled internal gauge error_2 {best_err:.9f} mean_local {best_local.mean():.9g}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, words=best_words, w=w.cpu().numpy(), tree_edges=edges, brot=np.array(args.brot))
    args.out.with_suffix(".json").write_text(json.dumps({"base_error_2": base_err, "sampled_error_2": best_err, "samples": args.samples, "mean_local_projector_error": float(best_local.mean())}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
