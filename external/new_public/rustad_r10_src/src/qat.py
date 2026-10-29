"""Quantization-aware refit of a dfthcsos factorization at a target brot (torch, CUDA, float64).

Free parameters: two-body angles th2 (R*B, N-1), one-body angles th1 (N, N-1), weights w (R,B,C), offsets Wo (R,C).
Angles enter through a straight-through round to the brot grid, so the loss is always that of the words that would
be submitted. Loss = lambda_eff + penalties on error_1, error_2 (Frobenius, as in the verifier) and the floors.

  python src/qat.py IN.npz OUT.npz --brot 7 --steps 4000
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dft_eval as de  # noqa: E402

N = 76
dev = torch.device("cuda")
DT = torch.float64


_LAYERS = {}


def _layers(edges, device):
    """Split the ordered edge list into runs of node-disjoint rotations (they commute within a run)."""
    key = (tuple(map(tuple, edges)), str(device))
    if key not in _LAYERS:
        layers, cur, touched = [], [], set()
        for p, q in edges:
            if p in touched or q in touched:
                layers.append(cur)
                cur, touched = [], set()
            cur.append((int(p), int(q)))
            touched.update((int(p), int(q)))
        if cur:
            layers.append(cur)
        _LAYERS[key] = [(torch.tensor([p for p, _ in L], device=device), torch.tensor([q for _, q in L], device=device),
                         len(L)) for L in layers]
    return _LAYERS[key]


def decode(th, edges):
    """theta/pi (M, N-1) -> unit vectors (M, N) with the verifier's Givens map (disjoint rotations batched)."""
    M = th.shape[0]
    ang = torch.pi * th
    c, s = torch.cos(ang), torch.sin(ang)
    v = torch.zeros(M, N, dtype=th.dtype, device=th.device)
    v[:, 0] = 1
    t = 0
    for P, Q, n in _layers(edges, th.device):
        cs, ss = c[:, t:t + n], s[:, t:t + n]
        a, b = v[:, P], v[:, Q]
        v = v.index_copy(1, P, cs * a - ss * b).index_copy(1, Q, ss * a + cs * b)
        t += n
    return v


def qround(th, brot):
    scale = float(1 << (brot - 1))
    q = torch.floor(th * scale + 0.5) / scale
    return th + (q - th).detach()


class Problem:
    def __init__(self, edges):
        r = de.ref()
        S, beta_1 = de.shifts()
        self.edges = [tuple(map(int, e)) for e in edges]
        tei = r["tei"].astype(np.float64)
        h1_thc = r["oei"] - 0.5 * np.einsum("prrq->pq", r["tei"])
        # packed (p>=q) representation; off-diagonal pairs weighted by sqrt(2) on each side
        il = np.tril_indices(N)
        self.il = il
        cw = np.where(il[0] == il[1], 1.0, np.sqrt(2.0))
        T = tei.copy()
        d = np.arange(N)
        T[:, :, d, d] += 0.5 * S[:, :, None]
        T[d, d, :, :] += 0.5 * S[None, :, :]
        Tp = T[il[0], il[1]][:, il[0], il[1]] * cw[:, None] * cw[None, :]
        self.Tp = torch.tensor(Tp, dtype=DT, device=dev)
        self.cw = torch.tensor(cw, dtype=DT, device=dev)
        self.il_t = (torch.tensor(il[0], device=dev), torch.tensor(il[1], device=dev))
        self.base1 = torch.tensor(h1_thc - 0.5 * de.ETA * S + beta_1 * np.eye(N), dtype=DT, device=dev)
        self.const_sos = float(r["ecore"] - de.ETA * beta_1)

    def metrics(self, th1, th2, w, Wo, brot, quant=True):
        R, B, C = w.shape
        t1 = qround(th1, brot) if quant else th1
        t2 = qround(th2, brot) if quant else th2
        v2 = decode(t2, self.edges).reshape(R, B, N)
        X = torch.einsum("rbp,rbq,rbc->rcpq", v2, v2, w)  # (R,C,N,N), symmetric
        A = X[:, :, self.il_t[0], self.il_t[1]].reshape(R * C, -1) * self.cw  # (RC, P)
        G = A.T @ A
        e2 = torch.linalg.norm(G - self.Tp)
        h1p = self.base1 - torch.einsum("rc,rcpq->pq", Wo, X)
        lam = torch.linalg.eigvalsh(h1p)  # only eigenvalues carry gradients (eigenvector grads blow up)
        q1 = decode(t1, self.edges)  # (N, N) rows
        # match rows to eigenvectors one-to-one on |overlap|, as the verifier does
        with torch.no_grad():
            _, vec = torch.linalg.eigh(h1p.detach())
            ov = (q1 @ vec).abs().cpu().numpy()
            from scipy.optimize import linear_sum_assignment
            ri, ci = linear_sum_assignment(-ov)
            perm = torch.tensor(ci[np.argsort(ri)], device=dev)
        lam_m = lam[perm]
        h1q = torch.einsum("j,jp,jq->pq", lam_m, q1, q1)
        e1 = torch.linalg.norm(h1q - h1p)
        lam1 = lam.abs().sum()
        wB = Wo + w.sum(dim=1)
        lam2 = 0.25 * ((w.abs().sum(dim=1) + wB.abs()) ** 2).sum()
        Lam = lam1 + lam2
        esos = self.const_sos - (lam1 - lam.sum()) - 0.5 * (Wo ** 2).sum()
        gap = de.E_UPPER - esos
        leff = torch.sqrt(gap * (2 * Lam - gap))
        return dict(e1=e1, e2=e2, Lam=Lam, gap=gap, leff=leff)


def words_from(th, brot):
    scale = 1 << (brot - 1)
    return (np.floor(np.mod(th, 2.0) * scale + 0.5).astype(np.int64)) % (1 << brot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--brot", type=int, default=7)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr-th", type=float, default=2e-3)
    ap.add_argument("--lr-w", type=float, default=2e-4)
    ap.add_argument("--target", type=float, default=0.0185)
    ap.add_argument("--gap-margin", type=float, default=0.02)
    ap.add_argument("--mu", type=float, default=2e4)
    ap.add_argument("--rho", type=float, default=1.0, help="weight of lambda_eff in the loss")
    ap.add_argument("--log", type=int, default=100)
    a = ap.parse_args()

    z = np.load(a.inp, allow_pickle=False)
    params = {k: np.array(z[k]) for k in z.files}
    edges = params["tree_edges"].astype(np.int64)
    w0, Wo0 = de.native(params)
    R, B, C = w0.shape
    th2 = de.tree_angles(params["U"].reshape(-1, N), edges)
    # one-body init: eigenvectors of h1' of the quantized two-body vectors
    words0 = de.encode_all(params["U"], w0, Wo0, edges, a.brot)
    m0 = de.evaluate(words0, w0, Wo0, edges, a.brot)
    print("rounded start:", {k: round(float(v), 6) for k, v in m0.items()}, flush=True)
    v2 = de.decode(words0[N:], edges, a.brot).reshape(R, B, N)
    X = de.dm.reconstruct_x(v2, w0)
    _, ev = np.linalg.eigh(de.h1_prime(w0, Wo0, X))
    th1 = de.tree_angles(ev.T, edges)

    P = Problem(edges)
    th1_t = torch.tensor(th1, dtype=DT, device=dev, requires_grad=True)
    th2_t = torch.tensor(th2, dtype=DT, device=dev, requires_grad=True)
    w_t = torch.tensor(w0, dtype=DT, device=dev, requires_grad=True)
    Wo_t = torch.tensor(Wo0, dtype=DT, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{"params": [th1_t, th2_t], "lr": a.lr_th}, {"params": [w_t, Wo_t], "lr": a.lr_w}])
    best, best_state = None, None
    t0 = time.time()
    for step in range(a.steps + 1):
        m = P.metrics(th1_t, th2_t, w_t, Wo_t, a.brot)
        pen = (torch.relu(m["e1"] - a.target) ** 2 + torch.relu(m["e2"] - a.target) ** 2) * a.mu
        pen = pen + a.mu * torch.relu(de.GAP_FLOOR + a.gap_margin - m["gap"]) ** 2
        pen = pen + a.mu * torch.relu(de.LAMBDA_FLOOR + 1.0 - m["Lam"]) ** 2
        loss = a.rho * m["leff"] + pen
        feas = (m["e1"].item() <= a.target + 5e-4 and m["e2"].item() <= a.target + 5e-4
                and m["gap"].item() > de.GAP_FLOOR + a.gap_margin / 2 and m["Lam"].item() > de.LAMBDA_FLOOR + 0.5)
        if feas and (best is None or m["leff"].item() < best):
            best = m["leff"].item()
            best_state = [t.detach().cpu().numpy().copy() for t in (th1_t, th2_t, w_t, Wo_t)]
        if step % a.log == 0:
            print(f"{step:6d} leff {m['leff'].item():.4f} e1 {1e3 * m['e1'].item():8.3f} e2 {1e3 * m['e2'].item():8.3f} "
                  f"gap {m['gap'].item():.4f} Lam {m['Lam'].item():.3f} best {best} {time.time() - t0:.0f}s", flush=True)
        if step == a.steps:
            break
        opt.zero_grad()
        loss.backward()
        grads = [t.grad for t in (th1_t, th2_t, w_t, Wo_t)]
        if any(not torch.isfinite(g).all() for g in grads):
            print(f"{step}: non-finite gradient, skipping step", flush=True)
            continue
        torch.nn.utils.clip_grad_norm_([th1_t, th2_t, w_t, Wo_t], 1e3)
        opt.step()

    if best_state is None:
        print("no feasible state found; saving last state for inspection")
        best_state = [t.detach().cpu().numpy().copy() for t in (th1_t, th2_t, w_t, Wo_t)]
    th1b, th2b, wb, Wob = best_state
    words = np.vstack([words_from(th1b, a.brot), words_from(th2b, a.brot)])
    m = de.evaluate(words, wb, Wob, edges, a.brot)
    print("verifier-math check:", {k: round(float(v), 6) for k, v in m.items()}, "OK" if de.ok(m) else "FAIL")
    U = de.decode(words[N:], edges, a.brot).reshape(R, B, N)
    out = dict(params)
    out.update(U=U, W=wb, H2_I=Wob + wb.sum(axis=1), angle_words=words, brot=np.array(a.brot))
    np.savez(a.out, **out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
