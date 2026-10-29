"""Optimize the exact Givens-tree sign gauge before lattice rounding.

Every edge angle has four equivalent representations because the child column
is zero when the edge is created: the parent and child output signs may be
chosen independently.  A tree dynamic program picks the equivalent
representation with the smallest predicted rounding error, while enforcing a
single final sign for the whole vector (the projector is unchanged).
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

import dfthc_torch as dt
import givens
import refdata
from gtrain import projector_design, refit_block, residual, xmat


def tree_children(edges: np.ndarray) -> list[list[tuple[int, int]]]:
    children = [[] for _ in range(len(edges) + 1)]
    for t, (p, c) in enumerate(edges):
        children[int(p)].append((t, int(c)))
    return children


def transformed_angle(theta: float, parent_in: int, parent_out: int, child_out: int) -> float:
    """Equivalent angle in units of pi for the requested output signs."""
    return float(np.arctan2(child_out * parent_in * np.sin(np.pi * theta),
                             parent_out * parent_in * np.cos(np.pi * theta)) / np.pi)


def circular_round_error(theta: float, scale: int) -> float:
    word = int(np.rint(theta * scale)) % (2 * scale)
    q = word / scale
    return float(((q - theta + 1.0) % 2.0) - 1.0)


def best_gauge(theta: np.ndarray, edges: np.ndarray, brot: int) -> tuple[np.ndarray, float, int]:
    children = tree_children(edges)
    scale = 1 << (brot - 1)
    n_nodes = len(children)

    def solve_one(row: np.ndarray, final_sign: int) -> tuple[np.ndarray, float]:
        choice: dict[tuple[int, int], list[tuple[int, int, int, float]]] = {}

        @lru_cache(None)
        def dp(node: int, initial_sign: int) -> float:
            states = {initial_sign: 0.0}
            records: list[dict[int, tuple[float, int, int, int, float]]] = []
            for edge_idx, child in children[node]:
                new: dict[int, float] = {}
                rec: dict[int, tuple[float, int, int, int, float]] = {}
                for current, base_cost in states.items():
                    for parent_out in (-1, 1):
                        for child_out in (-1, 1):
                            angle = transformed_angle(row[edge_idx], current, parent_out, child_out)
                            child_cost = dp(child, child_out)
                            if not np.isfinite(child_cost):
                                continue
                            cost = base_cost + child_cost + circular_round_error(angle, scale) ** 2
                            if cost < new.get(parent_out, float("inf")):
                                new[parent_out] = cost
                                rec[parent_out] = (cost, current, parent_out, child_out, angle)
                states = new
                records.append(rec)
            value = states.get(final_sign, float("inf"))
            # Store one backpointer per child position for this node/state.
            for end_sign in states:
                cur = end_sign
                out: list[tuple[int, int, int, float]] = []
                for rec in reversed(records):
                    item = rec[cur]
                    _, prev, pout, cout, angle = item
                    out.append((prev, pout, cout, angle))
                    cur = prev
                choice[(node, initial_sign, end_sign)] = list(reversed(out))
            return value

        cost = dp(0, 1)
        if not np.isfinite(cost):
            raise RuntimeError("no valid sign gauge")
        out = row.copy()

        def recover(node: int, initial_sign: int) -> None:
            states = choice[(node, initial_sign, final_sign)]
            current = initial_sign
            for (edge_idx, child), (prev, parent_out, child_out, angle) in zip(children[node], states):
                if prev != current:
                    raise AssertionError("gauge backpointer mismatch")
                out[edge_idx] = angle
                recover(child, child_out)
                current = parent_out

        recover(0, 1)
        return out, cost

    candidates = [solve_one(theta, sign) for sign in (-1, 1)]
    return min(candidates, key=lambda item: item[1])[0], min(candidates, key=lambda item: item[1])[1], 0


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
    parser.add_argument("--brot", type=int, default=8)
    args = parser.parse_args()

    problem = dt.Problem.build(refdata.load())
    with np.load(args.start, allow_pickle=False) as data:
        theta = np.asarray(data["theta"], dtype=np.float64)
        w = torch.tensor(np.asarray(data["w"]), dtype=torch.float32, device=problem.device)
        edges = np.asarray(data["tree_edges"], dtype=np.int64)
    base_words = givens.to_words(theta, args.brot)
    base_theta = givens.words_to_theta(base_words, args.brot)
    print(f"base nearest error_2 {evaluate(problem, base_theta, w, edges, args.brot):.9f}", flush=True)

    transformed = np.empty_like(theta)
    gauge_cost = 0.0
    for i, row in enumerate(theta):
        transformed[i], cost, _ = best_gauge(row, edges, args.brot)
        gauge_cost += cost
    words = givens.to_words(transformed, args.brot)
    quant_theta = givens.words_to_theta(words, args.brot)
    err = evaluate(problem, quant_theta, w, edges, args.brot)
    print(f"DP gauge error_2 {err:.9f} angle_cost {gauge_cost:.9g}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, words=words, w=w.cpu().numpy(), tree_edges=edges, brot=np.array(args.brot))
    args.out.with_suffix(".json").write_text(json.dumps({"base_error_2": evaluate(problem, base_theta, w, edges, args.brot), "dp_error_2": err, "angle_cost": gauge_cost}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
