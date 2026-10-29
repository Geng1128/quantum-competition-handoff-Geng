"""All-input phase refit of the attributed EXP222 arithmetic, without seed search.

Bit-plane tracing selects zero HMR outcomes. Before the final rotation the only
classical feedback is diagonal CZ, so all rotation feature bits are independent
of measurement outcomes. Existing arithmetic/uncomputation stays unchanged.
Every candidate is screened on the complete domain; this is still not the full
official randomized verifier. Coefficients are emitted on the exact 2^-64-turn grid.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linprog


MASK64 = (1 << 64) - 1
MAX_X = 3133452
BASELINE = Path(__file__).with_name("baseline") / "invsqrt.kmx"
BASELINE_SHA = "1d0e6d52eb79b8be4216bb276c22988c514f61e8dd06b3fb3edf8d9439ac4828"
SCALE = 1 << 20  # LP units: 2^-44 turns. Stored phase units: 2^-64 turns.


def half_turns_to_units(value: Fraction) -> int:
    """Official nearest 2^-64-turn rounding, modulo one turn."""
    value = Fraction(value)
    n, d = value.numerator, value.denominator
    return (((n << 64) // d + 1) >> 1) & MASK64


def reference_units(xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact beta=1, precision_bits=24 official reference and floored tolerance."""
    xs = np.asarray(xs)
    if xs.ndim != 1 or xs.dtype.kind not in "iu" or not xs.size:
        raise ValueError("Reference inputs must be a nonempty integer vector.")
    target = np.empty(len(xs), dtype=np.uint64)
    tolerance = np.empty_like(target)
    for i, x in enumerate(xs):
        if not 1 <= int(x) <= MAX_X:
            raise ValueError("Reference input outside official domain.")
        r = math.isqrt((1 << 130) // int(x))
        target[i] = ((r + 1) >> 1) & MASK64
        tolerance[i] = min(r >> 25, 1 << 63)
    return target, tolerance


def trace_features(body: str, xs: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Trace basis bits in packed uint64 planes through the last Z_POW.

    This intentionally rejects classical feedback that could change a later
    rotation feature. Measurements and reset clear the measured basis bit.
    """
    xs = np.asarray(xs)
    if (
        xs.ndim != 1
        or xs.dtype.kind not in "iu"
        or not xs.size
        or np.any(xs < 0)
        or np.any(xs >= 1 << 22)
    ):
        raise ValueError("Expected a nonempty 22-bit input vector.")
    xs = xs.astype(np.uint64)
    lines = body.splitlines()
    phase_lines = [i for i, line in enumerate(lines) if line.split()[:1] == ["Z_POW"]]
    if not phase_lines:
        raise ValueError("No phase instructions.")
    prefix = lines[: phase_lines[-1] + 1]
    wires = [
        int(t[1:])
        for line in prefix
        for t in line.split()
        if t.startswith("q") and t[1:].isdigit()
    ]
    count = len(xs)
    planes = np.zeros((max(22, max(wires) + 1), (count + 63) // 64), dtype=np.uint64)
    padded = np.zeros(planes.shape[1] * 64, dtype=np.uint64)
    padded[:count] = xs
    for q in range(22):
        planes[q] = np.packbits(
            ((padded >> q) & 1).astype(np.uint8), bitorder="little"
        ).view(np.uint64)
    features = np.empty((len(phase_lines), count), dtype=np.bool_)
    sites = []
    for line_number, line in enumerate(prefix, 1):
        tokens = line.split()
        if not tokens:
            continue
        op = tokens[0]
        if op in ("APPEND_TO_REGISTER", "REGISTER", "INFO_TAG"):
            continue
        if "if" in tokens:
            if (
                op != "CZ"
                or len(tokens) != 5
                or tokens[-2] != "if"
                or not tokens[-1].startswith("b")
            ):
                raise ValueError("Unsupported classical feedback in phase prefix.")
            continue  # Zero-outcome branch; diagonal feedback never changes bits.
        qs = [int(t[1:]) for t in tokens[1:] if t.startswith("q") and t[1:].isdigit()]
        if op == "X":
            planes[qs[0]] ^= np.uint64(MASK64)
        elif op == "CX":
            planes[qs[1]] ^= planes[qs[0]]
        elif op == "CCX":
            planes[qs[2]] ^= planes[qs[0]] & planes[qs[1]]
        elif op == "CSWAP":
            change = (planes[qs[1]] ^ planes[qs[2]]) & planes[qs[0]]
            planes[qs[1]] ^= change
            planes[qs[2]] ^= change
        elif op in ("R", "HMR"):
            planes[qs[0]] = 0
        elif op == "Z_POW":
            exponent = Fraction(tokens[2])
            features[len(sites)] = np.unpackbits(
                planes[qs[0]].view(np.uint8), bitorder="little"
            )[:count]
            sites.append(
                dict(
                    line=line_number,
                    wire=qs[0],
                    exponent=tokens[2],
                    phase_units=half_turns_to_units(exponent),
                    classification=(
                        "clifford"
                        if exponent % Fraction(1, 2) == 0
                        else "t" if exponent % Fraction(1, 4) == 0 else "rotation"
                    ),
                )
            )
        else:
            raise ValueError(f"Unsupported trace operation: {op}")
    return features, sites


def phase_units(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Accumulate dyadic phases modulo 2^64 without float conversion."""
    if features.ndim != 2 or coefficients.shape != (len(features),):
        raise ValueError("Phase feature / coefficient shape mismatch.")
    result = np.zeros(features.shape[1], dtype=np.uint64)
    for bits, value in zip(features, coefficients):
        result += bits.astype(np.uint64) * np.uint64(value)
    return result


def _float_adjustment(
    features: np.ndarray, indices: list[int], delta: np.ndarray
) -> np.ndarray:
    result = np.empty(features.shape[1], dtype=np.float64)
    for start in range(0, len(result), 32768):
        stop = min(start + 32768, len(result))
        # Bound the temporary float design matrix to <12 MiB.
        result[start:stop] = delta @ features[indices, start:stop].astype(np.float64)
    return result


def solve_refit(
    features: np.ndarray,
    base: np.ndarray,
    target: np.ndarray,
    tolerance: np.ndarray,
    variable_indices: list[int],
    *,
    dropped: list[int] | None = None,
    max_rounds: int = 40,
) -> dict:
    """Maximize a uniform absolute error margin using deterministic cutting planes."""
    features = np.asarray(features)
    if features.ndim != 2 or features.dtype != np.bool_ or not features.shape[1]:
        raise ValueError("Expected nonempty boolean phase features.")
    count = features.shape[1]
    normalized = []
    for values, name, length in (
        (base, "base", len(features)),
        (target, "target", count),
        (tolerance, "tolerance", count),
    ):
        values = np.asarray(values)
        if (
            values.shape != (length,)
            or values.dtype.kind not in "iu"
            or np.any(values < 0)
        ):
            raise ValueError(name + " must be a matching nonnegative integer vector.")
        normalized.append(values.astype(np.uint64, copy=False))
    base, target, tolerance = normalized
    if np.any(tolerance == 0):
        raise ValueError("Tolerance must be positive.")
    if len(set(variable_indices)) != len(variable_indices) or any(
        type(i) is not int or not 0 <= i < len(features) for i in variable_indices
    ):
        raise ValueError("Invalid phase variable indices.")
    if not set(dropped or []) <= set(variable_indices):
        raise ValueError("Dropped phases must be variables.")
    base_phase = phase_units(features, base)
    error = (base_phase - target).view(np.int64).astype(np.float64) / SCALE
    allowed = tolerance.astype(np.float64) / SCALE
    selected = set(np.linspace(0, count - 1, min(count, 2048), dtype=np.int64).tolist())
    selected.update(np.argsort(allowed)[: min(count, 512)].tolist())
    drop_set = set(dropped or [])
    bounds = [
        (
            (-int(base[i]) / SCALE, -int(base[i]) / SCALE)
            if i in drop_set
            else (-32768, 32768)
        )
        for i in variable_indices
    ]
    bounds.append((0, float(allowed.min())))
    objective = np.r_[np.zeros(len(variable_indices)), -1.0]
    history = []
    for iteration in range(max_rounds):
        rows = np.array(sorted(selected), dtype=np.int64)
        matrix = features[np.ix_(variable_indices, rows)].T.astype(np.float64)
        constraints = np.vstack(
            [
                np.column_stack([matrix, np.ones(len(rows))]),
                np.column_stack([-matrix, np.ones(len(rows))]),
            ]
        )
        rhs = np.r_[allowed[rows] - error[rows], allowed[rows] + error[rows]]
        solution = linprog(
            objective,
            A_ub=constraints,
            b_ub=rhs,
            bounds=bounds,
            method="highs",
            options={
                "primal_feasibility_tolerance": 1e-9,
                "dual_feasibility_tolerance": 1e-9,
            },
        )
        if not solution.success:
            return dict(
                status="INFEASIBLE" if solution.status == 2 else "SOLVER_FAILED",
                message=solution.message,
                dropped=sorted(drop_set),
                history=history,
                formal_verification=False,
            )
        delta, margin = solution.x[:-1], float(solution.x[-1])
        new_error = error + _float_adjustment(features, variable_indices, delta)
        margins = allowed - np.abs(new_error)
        violation = margin - margins
        worst = np.argpartition(violation, -min(count, 512))[-min(count, 512) :]
        history.append(
            dict(
                iteration=iteration,
                constraints_inputs=len(rows),
                lp_margin_units=margin,
                full_margin_units=float(margins.min()),
            )
        )
        if violation.max() <= 1e-6:
            coefficients = [int(v) for v in base]
            for i, change in zip(variable_indices, delta):
                coefficients[i] = (
                    coefficients[i] + round(float(change) * SCALE)
                ) & MASK64
            for i in drop_set:
                coefficients[i] = 0
            actual_phase = phase_units(
                features, np.array(coefficients, dtype=np.uint64)
            )
            actual_error = np.abs((actual_phase - target).view(np.int64)).astype(
                np.uint64
            )
            valid = bool(np.all(actual_error < tolerance))
            min_margin = int((tolerance - actual_error).min()) if valid else 0
            dual = (
                solution.ineqlin.marginals[: len(rows)]
                - solution.ineqlin.marginals[len(rows) :]
            )
            witness = [
                dict(input_index=int(index), weight=float(weight))
                for index, weight in zip(rows, dual)
                if abs(weight) > 1e-10
            ]
            return dict(
                status="MODEL_FULL_DOMAIN_PASS" if valid else "QUANTIZATION_FAILED",
                coefficients_units=coefficients,
                min_margin_units=min_margin,
                worst_error_over_tolerance=float(np.max(actual_error / tolerance)),
                dropped=sorted(drop_set),
                history=history,
                inputs=count,
                formal_verification=False,
                dual_witness=witness,
            )
        before = len(selected)
        selected.update(int(i) for i in worst if violation[i] > 1e-6)
        if len(selected) == before:
            return dict(
                status="NO_CUT_PROGRESS", history=history, formal_verification=False
            )
    return dict(status="ITERATION_LIMIT", history=history, formal_verification=False)


def rewrite_phases(body: str, sites: list[dict], coefficients: list[int]) -> str:
    """Only replace/remove attributed phase instructions; keep all other bytes."""
    if len(sites) != len(coefficients) or any(
        not isinstance(v, (int, np.integer))
        or isinstance(v, (bool, np.bool_))
        or not 0 <= int(v) <= MASK64
        for v in coefficients
    ):
        raise ValueError("Expected one valid phase-unit coefficient per site.")
    if len({s["line"] for s in sites}) != len(sites):
        raise ValueError("Phase sites must be unique.")
    replacements = dict(zip((s["line"] for s in sites), coefficients))
    lines = body.splitlines(keepends=True)
    for number, units in replacements.items():
        if type(number) is not int or not 1 <= number <= len(lines):
            raise ValueError("Invalid phase site line.")
        tokens = lines[number - 1].split()
        if tokens[0] != "Z_POW":
            raise ValueError("Phase site changed.")
        if units == 0:
            lines[number - 1] = ""
        else:
            with localcontext() as context:
                context.prec = 100
                exponent = format(Decimal(units) / Decimal(1 << 63), "f")
            lines[number - 1] = f"Z_POW {tokens[1]} {exponent}\n"
    return "".join(lines)


def project_cost(result: dict) -> dict:
    coefficients = result["coefficients_units"]
    rotations = sum(v != 0 and v % (1 << 61) != 0 for v in coefficients)
    t_gates = sum(v % (1 << 62) != 0 and v % (1 << 61) == 0 for v in coefficients)
    margin = result["min_margin_units"] / 2**64
    eps = math.pi * margin / rotations if rotations else None
    synthesis_t = rotations * (0.56 * math.log2(1 / eps) + 5.3) if rotations else 0
    cost = 82 * (1397 + (t_gates + synthesis_t) / 4)
    return dict(
        projected_cost=cost,
        num_rotations=rotations,
        num_t_gates=t_gates,
        projected_rotation_eps=eps,
        nominal_mean_toffoli=1397,
        scope="Deterministic all-input phase model plus nominal Toffoli expectation; not official verification.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drop-counts", nargs="+", type=int, default=[0, 1, 2, 3])
    args = parser.parse_args()
    body = BASELINE.read_bytes().decode()
    if hashlib.sha256(body.encode()).hexdigest() != BASELINE_SHA:
        raise ValueError("Baseline source hash changed.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    xs = np.arange(1, MAX_X + 1, dtype=np.uint64)
    started = time.monotonic()
    features, sites = trace_features(body, xs)
    print(
        f"Traced {len(xs)} inputs / {len(sites)} phase sites in {time.monotonic()-started:.1f}s",
        flush=True,
    )
    np.save(args.output_dir / "features.npy", features, allow_pickle=False)
    target, tolerance = reference_units(xs)
    base = np.array([s["phase_units"] for s in sites], dtype=np.uint64)
    indices = [i for i, s in enumerate(sites) if s["classification"] == "rotation"]
    smallest = sorted(indices, key=lambda i: int(base[i]))
    reports = []
    for number in args.drop_counts:
        if not 0 <= number <= len(indices):
            raise ValueError("Invalid drop count.")
        print(f"Refitting with {number} smallest rotations removed", flush=True)
        result = solve_refit(
            features, base, target, tolerance, indices, dropped=smallest[:number]
        )
        if result["status"] == "MODEL_FULL_DOMAIN_PASS":
            result.update(project_cost(result))
            candidate = rewrite_phases(body, sites, result["coefficients_units"])
            path = args.output_dir / f"drop_{number}.kmx"
            path.write_bytes(candidate.encode())
            result.update(
                candidate=str(path),
                assembly_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
            )
        reports.append(result)
        (args.output_dir / "results.json").write_text(
            json.dumps(
                dict(
                    source_sha256=BASELINE_SHA,
                    sites=sites,
                    candidates=reports,
                    official_verification="NOT_RUN",
                    seed_search_performed=False,
                ),
                indent=2,
            )
            + "\n"
        )
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in [
                        "status",
                        "projected_cost",
                        "num_rotations",
                        "min_margin_units",
                    ]
                    if k in result
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
