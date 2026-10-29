"""Differentiable DF-THC-SOS metrics in torch (GPU), mirroring the verifier.

Formulas follow ``dfthcsos/scoreinfra/dfthc_math.py``::

    X^(rho c)  = sum_b w_b^(rho c) u_b u_b^T             (u rows normalized)
    error_2    = || sum X (x) X - (S (x) I + I (x) S)/2 - tei ||_F
    h1_prime   = H1_THC - (eta/2) S + beta_1 I - sum W X
    Lambda     = sum |eig(h1_prime)| + 1/4 sum (sum_b |w| + |W + sum_b w|)^2
    E_SOS      = ecore - eta beta_1 - (sum|eig| - sum eig) - 1/2 sum W^2
    E_gap      = E_upper - E_SOS
    lambda_eff = sqrt(E_gap (2 Lambda - E_gap))

The one-body rows are assumed to be exact eigenvectors here (``error_1`` comes
only from angle quantization and is checked with the official code).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import refdata

torch.backends.cuda.matmul.fp32_precision = "ieee"


@dataclass(frozen=True)
class Problem:
    target: torch.Tensor  # (N^2, N^2) float32
    target_sq: float  # ||target||_F^2 in float64
    h1_base: torch.Tensor  # H1_THC - eta/2 S + beta_1 I, float64 (N, N)
    ecore_term: float  # ecore - eta * beta_1
    n: int
    device: torch.device

    @classmethod
    def build(cls, ref: refdata.Reference, device: str = "cuda") -> "Problem":
        n = ref.num_orbitals
        target64 = ref.two_body_target().reshape(n * n, n * n)
        target64 = 0.5 * (target64 + target64.T)
        h1_base = ref.h1_thc - 0.5 * ref.electrons * ref.S + ref.beta_1 * np.eye(n)
        dev = torch.device(device)
        return cls(
            target=torch.tensor(target64, dtype=torch.float32, device=dev),
            target_sq=float(np.sum(target64**2)),
            h1_base=torch.tensor(h1_base, dtype=torch.float64, device=dev),
            ecore_term=float(ref.ecore - ref.electrons * ref.beta_1),
            n=n,
            device=dev,
        )


def normalize_rows(U: torch.Tensor) -> torch.Tensor:
    return U / torch.linalg.vector_norm(U, dim=-1, keepdim=True)


def build_x(U: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``X[r, c] = sum_b w[r, b, c] u[r, b] u[r, b]^T`` -> (R, C, N, N)."""
    u = normalize_rows(U)
    return torch.einsum("rbp,rbq,rbc->rcpq", u, u, w)


def error2(problem: Problem, X: torch.Tensor) -> torch.Tensor:
    n2 = problem.n * problem.n
    xmat = X.reshape(-1, n2).to(torch.float32)
    residual = xmat.T @ xmat - problem.target
    return torch.linalg.vector_norm(residual)


def metrics(problem: Problem, U: torch.Tensor, w: torch.Tensor, Woff: torch.Tensor) -> dict[str, torch.Tensor]:
    """All scored metrics; ``U, w, Woff`` float64 tensors on ``problem.device``."""
    X = build_x(U, w)
    e2 = error2(problem, X)
    h1p = problem.h1_base - torch.einsum("rc,rcpq->pq", Woff, X)
    h1p = 0.5 * (h1p + h1p.T)
    eig = torch.linalg.eigvalsh(h1p)
    lam_h1 = torch.sum(torch.abs(eig))
    w_B = Woff + torch.sum(w, dim=1)
    lam_h2 = 0.25 * torch.sum(torch.square(torch.sum(torch.abs(w), dim=1) + torch.abs(w_B)))
    Lambda = lam_h1 + lam_h2
    e_sos = problem.ecore_term - (lam_h1 - torch.sum(eig)) - 0.5 * torch.sum(Woff**2)
    gap = refdata.UPPER_ENERGY - e_sos
    lam_eff = torch.sqrt(torch.clamp(gap * (2.0 * Lambda - gap), min=1e-12))
    return {
        "error_2": e2,
        "Lambda": Lambda,
        "lambda_h1": lam_h1,
        "lambda_h2": lam_h2,
        "E_SOS": e_sos,
        "E_gap": gap,
        "lambda_eff": lam_eff,
        "h1_prime": h1p,
    }


def scalar_report(m: dict[str, torch.Tensor]) -> dict[str, float]:
    return {k: float(v.detach().cpu()) for k, v in m.items() if v.ndim == 0}
