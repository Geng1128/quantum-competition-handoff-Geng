"""Progressive quantization with long L-BFGS compensation between freezes (angles + w, error_2 + D penalty).

Stage s freezes the not-yet-frozen angles with the largest rounding impact (|g d| + h d^2 / 2, as in pq.py) up to
fraction fracs[s], fixing them to their nearest grid value, then runs `iters` L-BFGS iterations on the free angles
and w. Frozen values are stored explicitly, so they cannot drift.

  python src/pq_lb.py IN.npz OUT.npz --brot 8 --fracs 0.02,0.05,0.1,0.2,0.4,0.7,1 --iters 300
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dft_eval as de  # noqa: E402
from qat import Problem, decode, dev, N  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--brot", type=int, default=8)
    ap.add_argument("--fracs", default="0.02,0.05,0.1,0.2,0.4,0.7,1")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--inner", type=int, default=25)
    ap.add_argument("--dpen", type=float, default=2e-5)
    ap.add_argument("--probes", type=int, default=4)
    ap.add_argument("--tol", type=float, default=0.0, help="stop a stage when 50 iterations gain < tol mHa")
    ap.add_argument("--first-iters", type=int, default=0, help="iteration cap for stage 1 (default: --iters)")
    ap.add_argument("--resume-frozen", default="", help="npz with frozen mask + values to start from")
    a = ap.parse_args()
    fracs = [float(f) for f in a.fracs.split(",")]
    z = np.load(a.inp, allow_pickle=False)
    params = {k: np.array(z[k]) for k in z.files}
    edges = params["tree_edges"].astype(np.int64)
    w0, Wo0 = de.native(params)
    R, B, C = w0.shape
    P = Problem(edges)
    dt = torch.float32
    Tp, cwt = P.Tp.to(dt), P.cw.to(dt)
    scale = float(1 << (a.brot - 1))
    if "th_cont" in params:
        th0 = params["th_cont"]
    else:
        th0 = de.tree_angles(params["U"].reshape(-1, N), edges)
    th = torch.tensor(th0, dtype=dt, device=dev, requires_grad=True)
    w = torch.tensor(w0, dtype=dt, device=dev, requires_grad=True)
    frozen = torch.zeros_like(th, dtype=torch.bool)
    qv = torch.zeros_like(th)
    if a.resume_frozen:
        zf = np.load(a.resume_frozen)
        frozen = torch.tensor(zf["frozen"], device=dev)
        qv = torch.tensor(zf["qv"], dtype=dt, device=dev)
        if "th" in zf.files:
            with torch.no_grad():
                th.copy_(torch.tensor(zf["th"], dtype=dt, device=dev))
                w.copy_(torch.tensor(zf["w"], dtype=dt, device=dev))

    def eff(t):
        return torch.where(frozen, qv, t)

    def e2fn(t, w_):
        v2 = decode(t, P.edges).reshape(R, B, N)
        M = (v2[:, :, P.il_t[0]] * v2[:, :, P.il_t[1]]) * cwt
        A = torch.einsum("rbc,rbp->rcp", w_, M).reshape(R * C, -1)
        return torch.linalg.norm(A.T @ A - Tp)

    def dfun(w_):
        return 0.5 * ((w_.abs().sum(1)) ** 2 - (w_.sum(1)) ** 2).sum()

    total = th.numel()
    t0 = time.time()
    for s, fr in enumerate(fracs, 1):
        target = int(round(total * fr))
        n_new = target - int(frozen.sum())
        if n_new > 0:
            t_eff = eff(th).detach().requires_grad_(True)
            f = e2fn(t_eff, w.detach()) ** 2
            g, = torch.autograd.grad(f, t_eff, create_graph=True)
            hd = torch.zeros_like(t_eff)
            for _ in range(a.probes):
                v = torch.randint(0, 2, t_eff.shape, device=dev).to(dt) * 2 - 1
                hv, = torch.autograd.grad(g, t_eff, grad_outputs=v, retain_graph=True)
                hd += v * hv / a.probes
            with torch.no_grad():
                qn = torch.floor(th * scale + 0.5) / scale
                d = qn - th
                impact = (g * d).abs() + 0.5 * hd.clamp(min=0) * d ** 2
                impact[frozen] = -1
                idx = torch.topk(impact.flatten(), n_new).indices
                frozen.view(-1)[idx] = True
                qv.view(-1)[idx] = qn.view(-1)[idx]
            del g, hd, f
        with torch.no_grad():
            e_frz = e2fn(eff(th), w).item()
        opt = torch.optim.LBFGS([th, w], lr=1, max_iter=a.inner, history_size=100, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-12, tolerance_change=1e-15)

        def closure():
            opt.zero_grad()
            loss = e2fn(eff(th), w) + a.dpen * dfun(w)
            loss.backward()
            th.grad[frozen] = 0
            return loss
        iters = (a.first_iters if (s == 1 and a.first_iters) else a.iters) if fr < 1 else 0
        last = None
        for it in range(iters):
            opt.step(closure)
            if it % 50 == 49:
                with torch.no_grad():
                    cur = 1e3 * e2fn(eff(th), w).item()
                print(f"   it {it + 1}: e2 {cur:.4f} mHa", flush=True)
                if last is not None and a.tol > 0 and last - cur < a.tol:
                    break
                last = cur
        with torch.no_grad():
            e = e2fn(eff(th), w).item()
        print(f"stage {s} frac {fr}: frozen {int(frozen.sum())} e2 after freeze {1e3 * e_frz:.4f} -> {1e3 * e:.4f} "
              f"mHa {time.time() - t0:.0f}s", flush=True)
        np.savez(a.out + ".state.npz", frozen=frozen.cpu().numpy(), qv=qv.cpu().numpy(),
                 th=th.detach().cpu().numpy(), w=w.detach().cpu().numpy(), stage=s)
        if fr >= 1:
            break
    if not bool(frozen.all()):
        print("not all frozen; state saved to", a.out + ".state.npz")
        return
    # final w-only L-BFGS
    optw = torch.optim.LBFGS([w], lr=1, max_iter=25, history_size=100, line_search_fn="strong_wolfe")

    def closure_w():
        optw.zero_grad()
        loss = e2fn(qv, w) + a.dpen * dfun(w)
        loss.backward()
        return loss
    for _ in range(40):
        optw.step(closure_w)
    with torch.no_grad():
        e = e2fn(qv, w).item()
    print(f"final (w refit) e2 {1e3 * e:.4f} mHa", flush=True)
    words2 = de.to_words(qv.cpu().numpy().astype(np.float64), a.brot)
    wn = w.detach().cpu().numpy().astype(np.float64)
    out = dict(params)
    out.update(U=de.decode(words2, edges, a.brot).reshape(R, B, N), W=wn, H2_I=Wo0 + wn.sum(axis=1), words2=words2,
               brot=np.array(a.brot))
    np.savez(a.out, **out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
