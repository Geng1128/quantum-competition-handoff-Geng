"""Recover EXP222's lookup and fit an independent fixed-point quadratic family.

The replay retains Phuc Thai's EXP222 assembly and its inherited attribution.
It reconstructs the lookup generator, not the unavailable EXP222 fitter.
The new quadratic fitter/model below is independent of that instruction stream.
Numerical screening is not official verification and does not check HMR branches.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


MAX_X = 3133452
BASELINE = Path(__file__).with_name("baseline") / "invsqrt.kmx"
BASELINE_SHA = "1d0e6d52eb79b8be4216bb276c22988c514f61e8dd06b3fb3edf8d9439ac4828"
LOOKUP_START, LOOKUP_STOP = 241, 6866  # Zero-based half-open line slice.
OUTPUT_ORDER = (
    [26]
    + list(range(45, 64))
    + list(range(44, 33, -1))
    + list(range(64, 72))
    + list(range(33, 26, -1))
)


def anf_coefficients(table: list[int]) -> list[int]:
    """Return exact vector-valued algebraic-normal-form coefficients."""
    if not table or len(table) & (len(table) - 1):
        raise ValueError("Lookup length must be a positive power of two.")
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in table):
        raise ValueError("Lookup words must be nonnegative Python integers.")
    result = list(table)
    for bit in range((len(table) - 1).bit_length()):
        for mask in range(len(table)):
            if mask & (1 << bit):
                result[mask] ^= result[mask ^ (1 << bit)]
    return result


def _authenticated_lines(body: str) -> list[str]:
    if hashlib.sha256(body.encode()).hexdigest() != BASELINE_SHA:
        raise ValueError(
            "Unexpected EXP222 assembly; fixed template boundaries are unsafe."
        )
    return body.splitlines(keepends=True)


def recover_lookup(body: str) -> list[int]:
    """Interpret the original 8-address-bit lookup on all 256 basis addresses.

    Choose HMR result zero. Its matching conditional phase correction vanishes;
    the output words are independent of this outcome. No notebook is executed.
    """
    lines = _authenticated_lines(body)
    state = np.zeros((79, 256), dtype=np.bool_)
    address = np.arange(256)
    for bit in range(8):
        state[14 + bit] = (address >> bit) & 1
    for line in lines[LOOKUP_START:LOOKUP_STOP]:
        tokens = line.split()
        op = tokens[0]
        if op == "CZ" and tokens[-2:] == ["if", "b0"]:
            continue
        wires = [int(q[1:]) for q in tokens[1:] if q.startswith("q")]
        if op == "X":
            state[wires[0]] ^= True
        elif op == "CX":
            state[wires[1]] ^= state[wires[0]]
        elif op == "CCX":
            state[wires[2]] ^= state[wires[0]] & state[wires[1]]
        elif op in ("R", "HMR"):
            state[wires[0]] = False
        else:
            raise ValueError(f"Unexpected lookup instruction: {line!r}")
    if state[72:].any():
        raise ValueError("Original lookup failed to clear its temporary ANDs.")
    return [
        sum(int(state[q, a]) << (q - 26) for q in range(26, 72)) for a in range(256)
    ]


def lookup_assembly(table: list[int]) -> str:
    """Rebuild the original DFS ANF lookup using six temporary AND qubits."""
    if len(table) != 256 or any(v >= 1 << 46 for v in table):
        raise ValueError("Expected 256 words of at most 46 bits.")
    coefficients = anf_coefficients(table)
    output = []

    def emit_word(mask: int, control: int | None) -> None:
        for q in OUTPUT_ORDER:
            if coefficients[mask] & (1 << (q - 26)):
                output.append(
                    f"X q{q}\n" if control is None else f"CX q{control} q{q}\n"
                )

    def visit(mask: int, control: int, last_bit: int, degree: int) -> None:
        emit_word(mask, control)
        for bit in range(last_bit + 1, 8):
            target = 71 + degree
            output.extend([f"R q{target}\n", f"CCX q{control} q{14 + bit} q{target}\n"])
            visit(mask | (1 << bit), target, bit, degree + 1)
            output.extend([f"HMR q{target} b0\n", f"CZ q{control} q{14 + bit} if b0\n"])

    emit_word(0, None)
    for bit in range(8):
        visit(1 << bit, 14 + bit, bit, 1)
    return "".join(output)


def replay_baseline(body: str, table: list[int]) -> str:
    """Reconstruct the lookup while retaining the exact arithmetic and erasure.

    A changed table requires changed uncomputation. Refuse to produce that
    unsafe combination; new coefficients belong in the independent generator.
    """
    lines = _authenticated_lines(body)
    if table != recover_lookup(body):
        raise ValueError("Changed coefficients need a new uncomputation circuit.")
    return (
        "".join(lines[:LOOKUP_START])
        + lookup_assembly(table)
        + "".join(lines[LOOKUP_STOP:])
    )


@dataclass(frozen=True)
class QuadraticConfig:
    address_bits: int = 8
    fraction_bits: int = 31
    slope_bits: int = 25
    curvature_bits: int = 15

    def __post_init__(self) -> None:
        for name, low, high in (
            ("address_bits", 3, 11),
            ("fraction_bits", 24, 36),
            ("slope_bits", 16, 30),
            ("curvature_bits", 8, 22),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}].")


def normalize_inputs(xs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized integers in [2^21,2^22), pair shifts, and odd shift."""
    xs = np.asarray(xs)
    if xs.dtype.kind not in "iu" or np.any(xs < 1) or np.any(xs > MAX_X):
        raise ValueError(f"Inputs must be integers in [1, {MAX_X}].")
    xs = xs.astype(np.int64)
    # Inputs are small integers, and log2 is exact at all power-of-two boundaries.
    shifts = (21 - np.floor(np.log2(xs)).astype(np.int64)) // 2
    paired = xs << (2 * shifts)
    parity = (paired < (1 << 21)).astype(np.int64)
    return paired << parity, shifts, parity


def _truncated_product(
    operand: np.ndarray,
    delta: np.ndarray,
    *,
    input_bits: int,
    output_bits: int,
    delta_bits: int,
) -> np.ndarray:
    """Sum individually truncated controlled rows, exactly matching the circuit."""
    result = np.zeros_like(delta, dtype=np.int64)
    for j in range(delta_bits):
        shift = output_bits - input_bits - 20 + j
        row = operand << shift if shift >= 0 else operand >> -shift
        result += ((delta >> j) & 1) * row
    return result


def _residual_product(
    delta: np.ndarray, b: np.ndarray, c: np.ndarray, config: QuadraticConfig
) -> np.ndarray:
    low_bits = 22 - config.address_bits
    slope = b - _truncated_product(
        c,
        delta,
        input_bits=config.curvature_bits,
        output_bits=config.slope_bits,
        delta_bits=low_bits,
    )
    if np.any(slope < 0):
        raise OverflowError("Quadratic slope became negative.")
    return _truncated_product(
        slope,
        delta,
        input_bits=config.slope_bits,
        output_bits=config.fraction_bits,
        delta_bits=low_bits,
    )


def fit_quadratic(config: QuadraticConfig) -> np.ndarray:
    """Fit positive coefficients for a - delta*(b - delta*c).

    Fit a quadratic on Chebyshev nodes; quantize curvature, then search a small
    slope neighborhood. Choose the constant from the discrete model's midrange
    residual on 65 integer nodes. Acceptance always uses the full input domain.
    """
    count = 1 << config.address_bits
    bins = count // 2
    low_bits = 22 - config.address_bits
    span = 1 << low_bits
    delta_grid = np.linspace(0, span - 1, 65).astype(np.int64)
    cheb_delta = (np.cos((np.arange(3) + 0.5) * np.pi / 3) + 1) * (span - 1) / 2**21
    table = []
    for address in range(count):
        factor = np.sqrt(2.0) if address >= bins else 1.0
        origin = 2.0 + (address % bins) * span / 2.0**20
        a, neg_b, c = np.polynomial.polynomial.polyfit(
            cheb_delta, factor / np.sqrt(origin + cheb_delta), 2
        )
        cq = round(c * 2**config.curvature_bits)
        center_b = round(-neg_b * 2**config.slope_bits)
        target = (
            factor / np.sqrt(origin + delta_grid / 2.0**20) * 2**config.fraction_bits
        )
        best = None
        for bq in range(max(0, center_b - 8), center_b + 9):
            product = _residual_product(delta_grid, np.int64(bq), np.int64(cq), config)
            needed = target + product
            aq = round(float((needed.max() + needed.min()) / 2))
            error = float(np.max(np.abs(aq - needed)))
            candidate = (error, abs(bq - center_b), aq, bq, cq)
            if best is None or candidate < best:
                best = candidate
        table.append(best[2:])
    return np.asarray(table, dtype=np.int64)


def quadratic_model(
    xs: np.ndarray, config: QuadraticConfig, table: np.ndarray
) -> np.ndarray:
    """Evaluate the column-truncated integer pipeline, returned in turns."""
    table = np.asarray(table)
    if table.shape != (1 << config.address_bits, 3) or table.dtype.kind not in "iu":
        raise ValueError("Invalid coefficient table shape or dtype.")
    limits = np.array(
        [
            1 << (config.fraction_bits + 1),
            1 << config.slope_bits,
            1 << config.curvature_bits,
        ]
    )
    if np.any(table < 0) or np.any(table >= limits):
        raise ValueError("A coefficient does not fit its register.")
    # Validated coefficients are below 2^37. Normalize unsigned input before
    # combining it with signed delta rows (NumPy otherwise promotes to float).
    table = table.astype(np.int64, copy=False)
    normalized, shifts, parity = normalize_inputs(xs)
    low_bits = 22 - config.address_bits
    addresses = ((normalized - (1 << 21)) >> low_bits) + (
        parity << (config.address_bits - 1)
    )
    delta = normalized & ((1 << low_bits) - 1)
    a, b, c = np.moveaxis(table[addresses], -1, 0)
    value = a - _residual_product(delta, b, c, config)
    if np.any(value < 0) or np.any(value >= 1 << (config.fraction_bits + 1)):
        raise OverflowError("Quadratic result does not fit its register.")
    return np.ldexp(value.astype(np.float64), shifts - 10 - config.fraction_bits)


def screen(config: QuadraticConfig, table: np.ndarray) -> dict:
    """Check every official input using the numerical model, with bounded memory."""
    worst, worst_x, min_margin = 0.0, 0, float("inf")
    for start in range(1, MAX_X + 1, 65536):
        xs = np.arange(start, min(start + 65536, MAX_X + 1), dtype=np.int64)
        target = 1 / np.sqrt(xs)
        error = np.abs(quadratic_model(xs, config, table) - target)
        tolerance = np.ldexp(target, -24)
        ratio = error / tolerance
        i = int(np.argmax(ratio))
        if ratio[i] > worst:
            worst, worst_x = float(ratio[i]), int(xs[i])
        min_margin = min(min_margin, float(np.min(tolerance - error)))
    return dict(
        status="MODEL_PASS" if worst < 1 else "MODEL_FAIL",
        inputs=MAX_X,
        worst_error_over_tolerance=worst,
        worst_x=worst_x,
        min_margin_turns=min_margin,
        formal_verification=False,
        scope="Float64 numerical screen of an exact integer pipeline; no HMR or synthesis verification.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--address-bits", type=int, default=8)
    parser.add_argument("--fraction-bits", type=int, default=31)
    parser.add_argument("--slope-bits", type=int, default=25)
    parser.add_argument("--curvature-bits", type=int, default=15)
    args = parser.parse_args()
    config = QuadraticConfig(
        args.address_bits, args.fraction_bits, args.slope_bits, args.curvature_bits
    )
    body = BASELINE.read_bytes().decode("utf-8")
    recovered = recover_lookup(body)
    replay = replay_baseline(body, recovered)
    if replay != body:
        raise ValueError("Lookup reconstruction did not reproduce the original bytes.")
    table = fit_quadratic(config)
    result = dict(
        config=asdict(config),
        baseline_replay_sha256=BASELINE_SHA,
        lookup_words=recovered,
        numerical_screen=screen(config, table),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "replayed.kmx").write_bytes(replay.encode())
    np.save(args.output_dir / "quadratic_coefficients.npy", table, allow_pickle=False)
    (args.output_dir / "reconstruction.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps({k: v for k, v in result.items() if k != "lookup_words"}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
