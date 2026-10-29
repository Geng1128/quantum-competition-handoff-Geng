"""Estimate ANGLE_k loader Toffolis for the cached-monomial + GF(2) low-rank scheme.

Address a = 2^L h + l. Per circuit: cache all nonlinear monomials of the L low
bits and of the high bits (one Toffoli each), then each output bit's table
T[h, l] (over valid addresses) is written as a sum of rank(T) products of two
linear forms, one Toffoli each. Identical tables share their terms.
"""

from __future__ import annotations

import argparse
import math

import numpy as np


def gf2_rank(rows: list[int]) -> int:
    rank = 0
    rows = [r for r in rows if r]
    while rows:
        pivot = rows.pop()
        if not pivot:
            continue
        rank += 1
        low = pivot & -pivot
        rows = [r ^ pivot if r & low else r for r in rows]
        rows = [r for r in rows if r]
    return rank


def signed_gauge(words: np.ndarray, edges: np.ndarray, brot: int) -> np.ndarray:
    """Exact integer gauge with every word in [-2^(brot-2), 2^(brot-2)) (mod 2^brot).

    Reverse tree order: a sign flip of a child's subtree negates its angle; a half
    turn (+pi) flips the parent amplitude sign, which propagates to the parent's angle.
    """
    mod, half, quarter = 1 << brot, 1 << (brot - 1), 1 << (brot - 2)
    out = words.astype(np.int64).copy()
    signs = np.ones((len(words), len(edges) + 1), dtype=np.int8)
    for t in range(len(edges) - 1, -1, -1):
        p, c = edges[t]
        ang = np.where(signs[:, p] != signs[:, c], (-out[:, t]) % mod, out[:, t])
        flip = (ang >= quarter) & (ang < 3 * quarter)
        ang = np.where(flip, (ang + half) % mod, ang)
        out[:, t] = ang
        signs[flip, p] *= -1
        signs[:, c] = 1
    return out


def circuit_cost(delta: np.ndarray, brot: int, low_bits: int, dedupe_sign: bool) -> int:
    A, J = delta.shape
    L = low_bits
    H = (A - 1 >> L) + 1
    low_cache = (1 << L) - L - 1
    high_bits = max(1, (H - 1).bit_length())
    high_cache = H - 1 - high_bits  # nonlinear monomials of the down-set [0, H)
    seen = set()
    total = 0
    bits = range(brot - 1) if dedupe_sign else range(brot)
    for j in range(J):
        for k in bits:
            col = (delta[:, j] >> k) & 1
            padded = np.zeros(H << L, dtype=np.int64)
            padded[:A] = col
            table = padded.reshape(H, 1 << L)
            rows = [int("".join(map(str, r[::-1])), 2) for r in table]
            key = tuple(rows)
            if key in seen or not any(rows):
                continue
            seen.add(key)
            total += gf2_rank(rows)
    return low_cache + max(0, high_cache) + total


def batch_deltas(words: np.ndarray, J: int) -> np.ndarray:
    A, n = words.shape
    padded = np.zeros((A, math.ceil(n / J) * J), dtype=np.int64)
    padded[:, :n] = words
    batches = padded.reshape(A, -1, J).transpose(1, 0, 2)
    out = batches.copy()
    out[1:] ^= batches[:-1]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--words", required=True, help=".npz with 'words' (A, 75)")
    parser.add_argument("--edges-from", required=True)
    parser.add_argument("--brot", type=int, default=8)
    parser.add_argument("--J", type=int, default=5)
    args = parser.parse_args()
    words = np.load(args.words)["words"].astype(np.int64)
    edges = np.load(args.edges_from)["tree_edges"].astype(np.int64)
    gauged = signed_gauge(words, edges, args.brot)
    for name, w in (("raw", words), ("signed", gauged)):
        deltas = batch_deltas(w, args.J)
        for L in (5, 6, 7):
            costs = [circuit_cost(d, args.brot, L, name == "signed") for d in deltas]
            print(f"{name} L={L}: per-circuit {costs} total/pass {sum(costs)}")


if __name__ == "__main__":
    main()
