"""Compile ANGLE_k circuits with output-mixing hubs, searching stage parameters per circuit.

Uses mrmeows' Apache-2.0 tensor compiler (alias_compiler/support: generic_hubs,
generic_lookup) as a library; this driver adds a per-circuit parameter search so
shapes other than its hard-coded L=23/24 can be compiled. Falls back to the
plain grouped decomposition when no hub schedule succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

SUPPORT = Path(__file__).resolve().parents[2] / "external/public_v2/_datasets/mrmeows_open/source/chemistry_research/alias_compiler/support"
sys.path.insert(0, str(SUPPORT))

from generic_hubs import factor_tensor, step  # noqa: E402
from generic_lookup import batch_deltas, build_circuit, check, decompose, prepare, signed_gauge  # noqa: E402


def candidates(L: int):
    """Hub schedules: (initial cap, stage-1 (nleft, nright, group, cap), stage-2 (...))."""
    for c0, s1, s2 in itertools.product(
        (L - 2, L - 3, L - 1),
        ((3, 0, 3, L - 11), (3, 0, 3, L - 12), (3, 0, 3, L - 13), (2, 0, 3, L - 12)),
        ((0, 3, 4, L - 8), (0, 3, 4, L - 9), (0, 3, 4, L - 10), (0, 2, 4, L - 9)),
    ):
        yield c0, s1, s2


def compile_hubs(delta: np.ndarray, k: int, max_tries: int):
    _, _, ac = prepare(delta, 8, 6)
    tensor = ac[1:, 1:]
    _, L, _ = tensor.shape
    best = None
    for i, (c0, s1, s2) in enumerate(candidates(L)):
        if i >= max_tries:
            break
        try:
            terms, _ = factor_tensor(tensor[:, :, :7], 7, c0)
            terms, _ = step(terms, tensor, 7, 19, *s1, seed=882 + k)
            terms, _ = step(terms, tensor, 19, 35, *s2, seed=1222 + k)
            circuit, predicted = build_circuit(delta, terms, 8, 6)
        except (ValueError, AssertionError):
            continue
        if best is None or predicted < best[1]:
            best = (circuit, predicted, {"c0": c0, "s1": s1, "s2": s2})
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--words", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--max-tries", type=int, default=48)
    args = parser.parse_args()
    src = np.load(args.submission)
    words, signs = signed_gauge(np.load(args.words), src["tree_edges"], 8)
    args.out.mkdir(parents=True, exist_ok=True)
    reports = []
    t0 = time.time()
    for k, delta in enumerate(batch_deltas(words, 5)[: args.count]):
        hub = compile_hubs(delta, k, args.max_tries)
        terms, _ = decompose(delta, 8, 6, 5, 17, 32)
        plain_c, plain_t = build_circuit(delta, terms, 8, 6)
        circuit, predicted, params = (hub if hub is not None and hub[1] < plain_t else (plain_c, plain_t, {"plain": True}))
        native = check(circuit, delta, brot=8, repeats=4, seed=63291 + k)
        assert native == predicted, (native, predicted)
        rel = args.out / f"angle_k_{k}.kmb"
        circuit.to_file(rel, format="kmb")
        rec = {"k": k, "T": native, "Q": int(circuit.num_qubits), "plain_T": plain_t, "hub_T": None if hub is None else hub[1], "params": params,
               "sha256": hashlib.sha256(rel.read_bytes()).hexdigest()}
        reports.append(rec)
        print(json.dumps(rec), f"{time.time() - t0:.0f}s", flush=True)
    np.save(args.out / "angle_words.npy", words)
    (args.out / "compile_manifest.json").write_text(json.dumps({"circuits": reports, "total_T_per_pass": sum(r["T"] for r in reports)}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
