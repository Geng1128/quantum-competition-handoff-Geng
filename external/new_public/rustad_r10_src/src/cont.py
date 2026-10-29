"""Long continuous refit of a DF-THC factorization (vectors U normalized in the forward pass, weights w) on error_2,
with an optional D(w) penalty (keeps squares near semidefinite, which keeps lambda_eff near its floor).

  python src/cont.py IN.npz OUT.npz --steps 50000 [--lr 2e-3 --lr-end 2e-5 --dpen 1e-4]
OUT keeps U continuous (float64) and the original Wo; quantize later with pq.py / flip.py.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dft_eval as de  # noqa: E402
from qat import Problem, dev, DT, N  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr-end", type=float, default=2e-5)
    ap.add_argument("--dpen", type=float, default=0.0)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--log", type=int, default=1000)
    ap.add_argument("--tau", type=float, default=0.0, help="final hard-threshold on |u| components (0 = off)")
    ap.add_argument("--tau-frac", type=float, default=0.5, help="fraction of steps over which tau ramps up")
    ap.add_argument("--tau-every", type=int, default=200)
    a = ap.parse_args()
    z = np.load(a.inp, allow_pickle=False)
    params = {k: np.array(z[k]) for k in z.files}
    edges = params["tree_edges"].astype(np.int64)
    w0, Wo0 = de.native(params)
    R, B, C = w0.shape
    P = Problem(edges)
    dt = torch.float32 if a.fp32 else DT
    Tp = P.Tp.to(dt)
    il0, il1, cw = P.il_t[0], P.il_t[1], P.cw.to(dt)
    U = torch.tensor(params["U"], dtype=dt, device=dev, requires_grad=True)
    w = torch.tensor(w0, dtype=dt, device=dev, requires_grad=True)
    opt = torch.optim.Adam([U, w], lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (a.lr_end / a.lr) ** (s / max(1, a.steps)))

    mask = torch.ones_like(U)

    def e2fn():
        Um = U * mask
        u = Um / Um.norm(dim=2, keepdim=True)
        M = u[:, :, il0] * u[:, :, il1] * cw
        A = torch.einsum("rbc,rbp->rcp", w, M).reshape(R * C, -1)
        return torch.linalg.norm(A.T @ A - Tp)

    best, best_state = 1e9, None
    t0 = time.time()
    for s in range(a.steps + 1):
        e = e2fn()
        dpen = 0.5 * ((w.abs().sum(1)) ** 2 - (w.sum(1)) ** 2).sum()
        loss = e + a.dpen * dpen
        if e.item() < best:
            best = e.item()
            if s % 100 == 0 or s == a.steps:
                best_state = ((U * mask).detach().cpu().numpy().astype(np.float64), w.detach().cpu().numpy().astype(np.float64))
        if a.tau > 0 and s % a.tau_every == 0 and s > 0:
            with torch.no_grad():
                tau = a.tau * min(1.0, s / (a.tau_frac * a.steps))
                Um = U * mask
                u = Um / Um.norm(dim=2, keepdim=True)
                keep = u.abs() >= tau
                top = u.abs().argmax(dim=2, keepdim=True)  # never empty a vector
                keep.scatter_(2, top, True)
                mask = (mask.bool() & keep).to(U.dtype)
                best = 1e9  # the support changed: restart best tracking
        if s % a.log == 0:
            nz = mask.sum(dim=2).mean().item()
            print(f"support {nz:.1f} ", end="")
            print(f"{s:7d} e2 {1e3 * e.item():.3f} mHa D {dpen.item():.4f} best {1e3 * best:.3f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{time.time() - t0:.0f}s", flush=True)
        if s == a.steps:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    Ub, wb = best_state
    Ub = Ub / np.linalg.norm(Ub, axis=2, keepdims=True)
    out = dict(params)
    for k in ("angle_words", "words1", "words2", "brot"):
        out.pop(k, None)
    out.update(U=Ub, W=wb, H2_I=Wo0 + wb.sum(axis=1))
    np.savez(a.out, **out)
    print("saved", a.out, "best e2 mHa", 1e3 * best)


if __name__ == "__main__":
    main()
