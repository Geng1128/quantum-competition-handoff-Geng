"""Re-optimize only (w, W) of a verified chemistry submission, keeping every angle word.

The circuits stay byte-identical; only ``submission.npz``'s ``W`` and ``H2_I``
change. Objective: lambda_eff subject to error_2, error_1 (with the fixed
one-body rows) and the E_gap floor, all with safety margins.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import dfthc_torch as dt
import givens
import refdata

if str(refdata.VERIFIERS_DIR) not in sys.path:
    sys.path.insert(0, str(refdata.VERIFIERS_DIR))
from dfthcsos.scoreinfra.circuits import load_circuits, recover_words_from_circuits  # noqa: E402
from dfthcsos.verify import native_inputs  # noqa: E402


def recover_words(src: Path, num_vectors: int) -> tuple[np.ndarray, int]:
    cache = src / "_recovered_words.npz"
    if cache.is_file():
        with np.load(cache) as data:
            return data["words"], int(data["brot"])
    # Native kickmix cannot open non-ASCII absolute paths; read relative to the directory.
    cwd = Path.cwd()
    try:
        import os

        os.chdir(src)
        circuits = load_circuits(Path("."))
    finally:
        os.chdir(cwd)
    words, layout, _ = recover_words_from_circuits(circuits, num_vectors=num_vectors, num_angles=75, shots_per_address=4, seed=1)
    np.savez(cache, words=words, brot=layout.brot)
    return words, layout.brot


def one_body_error(h1p: torch.Tensor, V: torch.Tensor, assign: torch.Tensor) -> torch.Tensor:
    vals = torch.linalg.eigvalsh(h1p)
    lam = vals[assign]  # eigenvalue matched to each one-body row
    rebuilt = torch.einsum("j,jp,jq->pq", lam, V, V)
    return torch.linalg.vector_norm(rebuilt - h1p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="chemistry dir with angle_k_*.kmb + submission.npz")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--e2", type=float, default=0.019985)
    parser.add_argument("--e1", type=float, default=0.0195)
    parser.add_argument("--gap-margin", type=float, default=2e-5)
    parser.add_argument("--w-lr-ratio", type=float, default=0.01)
    parser.add_argument("--penalty-scale", type=float, default=1e-3)
    args = parser.parse_args()

    with np.load(args.src / "submission.npz", allow_pickle=False) as data:
        params = {k: np.array(data[k]) for k in data.files}
    w0, W0, _, _ = native_inputs(params)
    R, B, C = w0.shape
    edges = params["tree_edges"].astype(np.int64)
    words, brot = recover_words(args.src, 76 + R * B)
    vectors = givens.decode_numpy(givens.words_to_theta(words, brot), edges)

    problem = dt.Problem.build(refdata.load())
    dev = problem.device
    V = torch.tensor(vectors[:76], dtype=torch.float64, device=dev)
    U = torch.tensor(vectors[76:].reshape(R, B, 76), dtype=torch.float64, device=dev)
    w = torch.tensor(w0, dtype=torch.float64, device=dev, requires_grad=True)
    Woff = torch.tensor(W0, dtype=torch.float64, device=dev, requires_grad=True)

    with torch.no_grad():
        m = dt.metrics(problem, U, w, Woff)
        _, vecs = torch.linalg.eigh(m["h1_prime"])
        overlaps = np.abs(V.cpu().numpy() @ vecs.cpu().numpy())
        rows, cols = linear_sum_assignment(-overlaps)
        assign = torch.tensor(cols[np.argsort(rows)], device=dev)
        e1 = float(one_body_error(m["h1_prime"], V, assign))
    start = {**dt.scalar_report(m), "error_1": e1}
    print("start", json.dumps(start), flush=True)

    gap_t = refdata.E_GAP_FLOOR + args.gap_margin
    opt = torch.optim.Adam([{"params": [w], "lr": args.lr * args.w_lr_ratio}, {"params": [Woff], "lr": args.lr}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.02)
    scale = args.penalty_scale
    best = None
    t0 = time.time()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        m = dt.metrics(problem, U, w, Woff)
        e1 = one_body_error(m["h1_prime"], V, assign)
        pen = (torch.relu(m["error_2"] - args.e2) / scale) ** 2 + (torch.relu(e1 - args.e1) / scale) ** 2 + (torch.relu(gap_t - m["E_gap"]) / scale) ** 2
        loss = m["lambda_eff"] + pen
        ok = float(m["error_2"]) <= args.e2 and float(e1) <= args.e1 and float(m["E_gap"]) >= gap_t
        lam = float(m["lambda_eff"].detach())
        if ok and (best is None or lam < best[0]):
            best = (lam, w.detach().clone(), Woff.detach().clone(), float(e1), float(m["error_2"]), float(m["E_gap"]))
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == args.steps - 1:
            print(f"step {step} lam_eff {lam:.5f} e2 {float(m['error_2']):.6f} e1 {float(e1):.6f} gap {float(m['E_gap']):.5f} best {best[0] if best else None} {time.time() - t0:.0f}s", flush=True)
    if best is None:
        raise SystemExit("no feasible point found")
    lam, w_b, W_b, e1_b, e2_b, gap_b = best
    args.out.mkdir(parents=True, exist_ok=True)
    for p in args.src.glob("angle_k_*.kmb"):
        shutil.copyfile(p, args.out / p.name)
    new = dict(params)
    w_np, W_np = w_b.cpu().numpy(), W_b.cpu().numpy()
    new["W"] = w_np.astype(params["W"].dtype)
    new["H2_I"] = (W_np + w_np.sum(axis=1)).astype(params["H2_I"].dtype)
    np.savez(args.out / "submission.npz", **new)
    summary = {"start": start, "best": {"lambda_eff": lam, "error_1": e1_b, "error_2": e2_b, "E_gap": gap_b}}
    (args.out / "polish_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
