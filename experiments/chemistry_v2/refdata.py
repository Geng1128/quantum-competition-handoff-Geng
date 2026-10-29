"""Cached FeMoCo76o113e reference data for chemistry experiments.

Loads the official verifier's reference (FCIDUMP + fixed symmetry shift) once
and caches the arrays in an ``.npz`` so later experiments start in seconds.
The official loader is used so that float32 rounding of ``tei`` and ``S``
matches the verifier exactly.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERIFIERS_DIR = PROJECT_ROOT / "external" / "competition" / "verifiers"
DEFAULT_CACHE = PROJECT_ROOT / "runs" / "chem_v2" / "ref_cache.npz"
UPPER_ENERGY = -22140.27335
E_GAP_FLOOR = 3.914898
ERROR_THRESHOLD = 0.02


@dataclass(frozen=True)
class Reference:
    oei: np.ndarray
    tei: np.ndarray
    h1_thc: np.ndarray
    S: np.ndarray
    beta_1: float
    ecore: float
    electrons: int

    @property
    def num_orbitals(self) -> int:
        return int(self.oei.shape[0])

    def two_body_target(self) -> np.ndarray:
        """``tei + (S (x) I + I (x) S) / 2``: what ``sum X (x) X`` must match."""
        n = self.num_orbitals
        target = self.tei.astype(np.float64).copy()
        diag = np.arange(n)
        target[:, :, diag, diag] += 0.5 * self.S[:, :, None]
        target[diag, diag, :, :] += 0.5 * self.S[None, :, :]
        return target


def _build_from_verifier() -> Reference:
    if str(VERIFIERS_DIR) not in sys.path:
        sys.path.insert(0, str(VERIFIERS_DIR))
    from dfthcsos.scoreinfra.dfthc_math import symmetric_from_lower
    from dfthcsos.scoreinfra.reference import check_fcidump, load_fixed_shifts, load_reference

    ref = load_reference(check_fcidump())
    shifts = load_fixed_shifts()
    n = int(ref["orbitals"])
    S = np.asarray(symmetric_from_lower(shifts["Hsym_lower"], n), dtype=np.float64)
    beta_1 = float(shifts["H1_I"]) - 0.5 * float(np.trace(S))
    return Reference(
        oei=np.asarray(ref["oei"], dtype=np.float64),
        tei=np.asarray(ref["tei"], dtype=np.float32),
        h1_thc=np.asarray(ref["h1_thc"], dtype=np.float64),
        S=S,
        beta_1=beta_1,
        ecore=float(ref["ecore"]),
        electrons=int(ref["electrons"]),
    )


def load(cache: Path = DEFAULT_CACHE) -> Reference:
    """Return the reference, building and caching it on first use."""
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as data:
            return Reference(
                oei=data["oei"],
                tei=data["tei"],
                h1_thc=data["h1_thc"],
                S=data["S"],
                beta_1=float(data["beta_1"]),
                ecore=float(data["ecore"]),
                electrons=int(data["electrons"]),
            )
    ref = _build_from_verifier()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        oei=ref.oei,
        tei=ref.tei,
        h1_thc=ref.h1_thc,
        S=ref.S,
        beta_1=ref.beta_1,
        ecore=ref.ecore,
        electrons=ref.electrons,
    )
    return ref
