"""Score model for chemistry shapes, using the official heuristic cost code.

The ANGLE_k loader cost is modelled after the best public loader
(mrmeows "4-address" banked QROM): ``ceil(A / banks) + 4`` Toffolis per
circuit per pass, ``K = ceil(75 / J)`` circuits, two passes.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from dataclasses import dataclass

import refdata

if str(refdata.VERIFIERS_DIR) not in sys.path:
    sys.path.insert(0, str(refdata.VERIFIERS_DIR))
from dfthcsos.verify import heuristic_costs  # noqa: E402

N = 76
NUM_ANGLES = N - 1


@dataclass(frozen=True)
class ShapeCost:
    R: int
    B: int
    C: int
    brot: int
    J: int
    toffoli: int
    qubits: int
    angles_toffoli: int
    heuristic_toffoli: int

    def score(self, lambda_eff: float) -> float:
        return lambda_eff * self.toffoli * self.qubits


def angles_toffoli(R: int, B: int, J: int, banks: int = 4, overhead: int = 4) -> int:
    addresses = N + R * B
    circuits = math.ceil(NUM_ANGLES / J)
    return 2 * circuits * (math.ceil(addresses / banks) + overhead)


def shape_cost(R: int, B: int, C: int, brot: int, J: int, angles_temp: int = 75) -> ShapeCost:
    heur = heuristic_costs(N, R, B, C, brot)
    ang = angles_toffoli(R, B, J)
    persistent = heur["persistent_qubits"] + J * brot
    temporary = max(heur["temporary_qubits"], angles_temp)
    return ShapeCost(R, B, C, brot, J, ang + heur["toffoli"], persistent + temporary, ang, heur["toffoli"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-eff", type=float, default=47.9)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    reference = shape_cost(4, 333, 128, 8, 5)
    print("mrmeows R4/B333/C128 model:", reference, f"score={reference.score(47.938):.4g}")
    rows = []
    for R, B, C, brot, J in itertools.product(
        (1, 2, 3, 4), range(100, 601, 20), range(100, 601, 20), (7, 8, 9, 10), (4, 5, 6, 7)
    ):
        if R * C < 290:
            continue
        if 11 + J * brot + 75 > 128:
            continue
        cost = shape_cost(R, B, C, brot, J)
        rows.append(cost)
    rows.sort(key=lambda c: c.toffoli * c.qubits)
    for c in rows[: args.top]:
        print(c, f"TQ={c.toffoli * c.qubits:.4g} score@{args.lambda_eff}={c.score(args.lambda_eff):.4g}")


if __name__ == "__main__":
    main()
