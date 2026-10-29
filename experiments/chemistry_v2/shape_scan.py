"""Scan (R, B, C) at brot 8 with the current public loader rate (J=5, ~0.235 Toffoli/address/circuit).

Calibrated on SomeJoint R8/B143/C62 (4333 loader Toffolis per pass, T 16752, Q 430).
"""

from __future__ import annotations

import argparse
import math
import sys

import refdata

if str(refdata.VERIFIERS_DIR) not in sys.path:
    sys.path.insert(0, str(refdata.VERIFIERS_DIR))
from dfthcsos.verify import heuristic_costs  # noqa: E402

N = 76
CIRCUITS = 15  # J = 5 words per ANGLE circuit, 75 angles
RATE = 4333 / (15 * (N + 8 * 143))


def cost(R: int, B: int, C: int, brot: int = 8) -> tuple[int, int, int]:
    heur = heuristic_costs(N, R, B, C, brot)
    loader = round(2 * CIRCUITS * RATE * (N + R * B))
    T = loader + heur["toffoli"]
    Q = heur["persistent_qubits"] + 5 * brot + max(heur["temporary_qubits"], 75)
    return T, Q, loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lam", type=float, default=48.27)
    args = parser.parse_args()
    T0, Q0, L0 = cost(8, 143, 62)
    print(f"SomeJoint R8/B143/C62 model: T {T0} Q {Q0} loader {L0} cost {args.lam * T0 * Q0:.4e} (actual 3.4772e8)")
    rows = []
    for R in range(4, 13):
        for B in range(76, 260, 1):
            for C in range(24, 130, 2):
                if R * C < 290:
                    continue
                T, Q, _ = cost(R, B, C)
                rows.append((T * Q, R, B, C, T, Q))
    rows.sort()
    seen = set()
    for tq, R, B, C, T, Q in rows:
        key = (R, C)
        if key in seen:
            continue
        seen.add(key)
        print(f"R{R} B{B} C{C}: T {T} Q {Q} TQ/TQ0 {tq / (T0 * Q0):.4f} params RBC={R * B * C}")
        if len(seen) >= 40:
            break


if __name__ == "__main__":
    main()
