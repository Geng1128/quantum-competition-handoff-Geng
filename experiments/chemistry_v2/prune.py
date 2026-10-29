"""Shrink an 8-bit factorization while staying on the word lattice.

Each step deletes the cheapest vectors (or copies) of every group, refits w
exactly, then repairs with discrete local search until error_2 is back under
``--target`` (or the sweep budget runs out). Checkpoints are saved per step.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import dfthc_torch as dt
import refdata
from dsearch import Search

if str(refdata.VERIFIERS_DIR) not in sys.path:
    sys.path.insert(0, str(refdata.VERIFIERS_DIR))
from dfthcsos.verify import native_inputs  # noqa: E402


def load_start(src: Path, device) -> tuple[np.ndarray, torch.Tensor, np.ndarray, int]:
    if (src / "submission.npz").is_file():
        with np.load(src / "submission.npz", allow_pickle=False) as data:
            params = {k: np.array(data[k]) for k in data.files}
        w, _, _, _ = native_inputs(params)
        with np.load(src / "_recovered_words.npz") as data:
            words, brot = np.array(data["words"])[76:], int(data["brot"])
        edges = params["tree_edges"].astype(np.int64)
    else:  # a checkpoint written by this script or dsearch.py
        with np.load(src, allow_pickle=False) as data:
            words, w, edges, brot = np.array(data["words"]), np.array(data["w"]), np.array(data["tree_edges"]), int(data["brot"])
    return words.astype(np.int64), torch.tensor(w, dtype=torch.float64, device=device), edges.astype(np.int64), brot


def repair(s: Search, target: float, max_sweeps: int, max_moves: int, rng) -> float:
    err = s.error()
    for _ in range(max_sweeps):
        if err <= target:
            break
        s.sweep((1, -1), max_moves, rng.permutation(s.R * s.B))
        s.refit(1)
        err = s.error()
    return err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--target-B", type=int, required=True)
    parser.add_argument("--target-C", type=int, default=None)
    parser.add_argument("--step", type=int, default=4, help="vectors removed per group per step")
    parser.add_argument("--c-step", type=int, default=4)
    parser.add_argument("--target", type=float, default=0.0195)
    parser.add_argument("--max-sweeps", type=int, default=3)
    parser.add_argument("--max-moves", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    problem = dt.Problem.build(refdata.load())
    words, w, edges, brot = load_start(args.start, problem.device)
    s = Search(problem, words, w, edges, brot)
    s.refit(2)
    rng = np.random.default_rng(0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = [{"R": s.R, "B": s.B, "C": s.C, "error_2": s.error()}]
    print(f"start R{s.R} B{s.B} C{s.C} error_2 {s.error():.6f}", flush=True)
    t0 = time.time()

    def checkpoint(tag: str, err: float) -> None:
        extra = {"copy_index": s.copy_index} if hasattr(s, "copy_index") else {}
        np.savez(args.out_dir / f"{tag}.npz", words=s.words.cpu().numpy(), w=s.w.cpu().numpy(), tree_edges=edges, brot=np.array(brot), **extra)
        log.append({"tag": tag, "R": s.R, "B": s.B, "C": s.C, "error_2": err, "seconds": time.time() - t0})
        (args.out_dir / "log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
        print(f"{tag}: R{s.R} B{s.B} C{s.C} error_2 {err:.6f} ({time.time() - t0:.0f}s)", flush=True)

    while s.B > args.target_B:
        s.remove_vectors(min(args.step, s.B - args.target_B))
        s.refit(1)
        removed = s.error()
        err = repair(s, args.target, args.max_sweeps, args.max_moves, rng)
        print(f"  B->{s.B}: after removal+refit {removed:.6f}, repaired {err:.6f}", flush=True)
        checkpoint(f"B{s.B}_C{s.C}", err)
    if args.target_C is not None:
        while s.C > args.target_C:
            s.remove_copies(min(args.c_step, s.C - args.target_C))
            s.refit(1)
            removed = s.error()
            err = repair(s, args.target, args.max_sweeps, args.max_moves, rng)
            print(f"  C->{s.C}: after removal+refit {removed:.6f}, repaired {err:.6f}", flush=True)
            checkpoint(f"B{s.B}_C{s.C}", err)


if __name__ == "__main__":
    main()
