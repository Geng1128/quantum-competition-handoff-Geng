"""Compile and sample-check the independent quadratic family with official Kickmix.

EXP222's authenticated normalization/uncomputation blocks retain their upstream
Phuc Thai / Dmitrii Gluzdov lineage. The polynomial arithmetic here is a new,
conservative reference generator, with explicit reversible row additions.
Local Windows checks are development evidence, not official Kaggle verification.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path

import kickmix as km
import numpy as np

from reconstruct import (
    BASELINE,
    BASELINE_SHA,
    MAX_X,
    QuadraticConfig,
    fit_quadratic,
    quadratic_model,
    screen,
)


def _add_product(
    builder,
    source,
    delta,
    target,
    input_bits,
    output_bits,
    *,
    inverse,
    arithmetic="explicit",
):
    """Add/subtract individually truncated rows; source and delta stay unchanged."""
    order = reversed(range(len(delta))) if inverse else range(len(delta))
    for j in order:
        shift = output_bits - input_bits - 20 + j
        terms = [
            (source[k], k + shift)
            for k in range(len(source))
            if 0 <= k + shift < len(target)
        ]
        if not terms:
            continue
        low = min(index for _, index in terms)
        if arithmetic == "native":
            sources = km.array([q for q, _ in terms])
            operation = builder.iadd if inverse else builder.isub
            operation(sources, target=target[low:], control=delta[j])
            continue
        products = builder.alloc_qubits(len(target) - low)
        sources = km.array([q for q, _ in terms])
        active_products = km.array([products[index - low] for _, index in terms])
        builder.init_and(delta[j], sources, active_products)
        if inverse:
            builder.iadd(products, target=target[low:])
        else:
            builder.isub(products, target=target[low:])
        builder.del_and(delta[j], sources, active_products)
        builder.free(products)


def build_quadratic(
    config: QuadraticConfig,
    table: np.ndarray,
    *,
    arithmetic: str = "explicit",
    max_qubits: int | None = None,
) -> tuple[str, dict]:
    """Build a separate circuit with a matching lookup and erasure for its table."""
    # Share validation with the model, before packing integer words or allocating.
    if arithmetic not in ("explicit", "native"):
        raise ValueError("Unknown arithmetic backend.")
    if max_qubits is not None and (type(max_qubits) is not int or max_qubits < 26):
        raise ValueError("max_qubits must be an integer >=26.")
    quadratic_model(np.array([1], dtype=np.int64), config, table)
    original = BASELINE.read_bytes()
    if hashlib.sha256(original).hexdigest() != BASELINE_SHA:
        raise ValueError("Unauthenticated normalization template.")
    lines = original.decode().splitlines(keepends=True)
    builder = km.CircuitBuilder()
    if max_qubits is not None:
        builder.set_max_qubits(max_qubits)
    reserved = builder.alloc_qubits(26)
    x, flags = reserved[:22], reserved[22:]
    low_bits = 22 - config.address_bits
    address, delta = x[low_bits:], x[:low_bits]
    widths = [int(table[:, i].max()).bit_length() for i in range(3)]
    # Result may exceed the initial constants during invalid configurations;
    # allocate the full model width and let screening reject those candidates.
    widths[0] = config.fraction_bits + 1
    registers = [builder.alloc_qubits(width) for width in widths]
    a, b, c = registers
    combined = km.array([q for register in registers for q in register])
    shifts = [0, widths[0], widths[0] + widths[1]]
    words = [
        sum(int(value) << shift for value, shift in zip(row, shifts)) for row in table
    ]
    builder.init_lookup(table=words, address=address, target=combined)
    _add_product(
        builder,
        c,
        delta,
        b,
        config.curvature_bits,
        config.slope_bits,
        inverse=False,
        arithmetic=arithmetic,
    )
    _add_product(
        builder,
        b,
        delta,
        a,
        config.slope_bits,
        config.fraction_bits,
        inverse=False,
        arithmetic=arithmetic,
    )

    extension = builder.alloc_qubits(10)
    y = km.array(list(a) + list(extension))
    # Valid normalized results have at most fraction_bits+1 bits. At most ten
    # leading zero pairs occur, so a cyclic shift is exactly a logical shift.
    builder.left_rotate(y, flags)
    denominator_bits = config.fraction_bits + 10
    for j, qubit in enumerate(y):
        exponent = Fraction(2) ** (j - denominator_bits + 1)
        if exponent % 2:
            builder.z_pow(qubit, exponent)
    builder.right_rotate(y, flags)
    builder.free(extension)

    _add_product(
        builder,
        b,
        delta,
        a,
        config.slope_bits,
        config.fraction_bits,
        inverse=True,
        arithmetic=arithmetic,
    )
    _add_product(
        builder,
        c,
        delta,
        b,
        config.curvature_bits,
        config.slope_bits,
        inverse=True,
        arithmetic=arithmetic,
    )
    builder.del_lookup(table=words, address=address, target=combined)
    builder.free(combined)
    body = builder.finish_circuit()
    buffer = io.BytesIO()
    body.to_file(buffer, format="kmx")
    polynomial = buffer.getvalue().decode()
    # These original blocks only require q0..q25 to remain unchanged in the
    # middle. Their temporary qubits and classical bits are clean/reusable.
    assembly = "".join(lines[:241]) + polynomial + "".join(lines[37330:])
    circuit = km.Circuit.from_file(io.BytesIO(assembly.encode()), format="kmx")
    if max_qubits is not None and circuit.num_qubits > max_qubits:
        raise ValueError(
            "Final circuit exceeds the qubit budget, including normalization."
        )
    body_stats = body.operation_statistics()
    stats = circuit.operation_statistics()
    metadata = dict(
        config=asdict(config),
        arithmetic=arithmetic,
        max_qubits_budget=max_qubits,
        coefficient_register_widths=widths,
        num_qubits=circuit.num_qubits,
        static_toffoli_like=int(stats.num_toffoli_like),
        polynomial_toffoli_like=int(body_stats.num_toffoli_like),
        num_ops=len(circuit),
        # All polynomial CCX/CSWAP are unconditional. Omitting normalization and
        # every T/rotation makes this a genuine lower bound on official Q*C.
        cost_lower_bound=circuit.num_qubits * int(body_stats.num_toffoli_like),
        assembly_sha256=hashlib.sha256(assembly.encode()).hexdigest(),
        normalization_source_sha256=BASELINE_SHA,
        formal_verification=False,
    )
    return assembly, metadata


def check_native(
    circuit: km.Circuit, xs: np.ndarray, expected: np.ndarray, *, seed: int = 1
) -> dict:
    """Sample-check all scratch, restored input, and tracked phase against the model."""
    xs = np.asarray(xs)
    expected = np.asarray(expected, dtype=np.float64)
    if (
        expected.shape != xs.shape
        or expected.ndim != 1
        or not np.isfinite(expected).all()
    ):
        raise ValueError(
            "expected must be a finite phase vector matching the input shape."
        )
    if not xs.size or xs.dtype.kind not in "iu" or np.any(xs < 1) or np.any(xs > MAX_X):
        raise ValueError(f"xs must be a nonempty integer vector in [1, {MAX_X}].")
    xs = xs.astype(np.uint64, copy=False)
    sim = km.Simulator(batch_size=len(xs), seed=seed, counting_operations=True)
    x = km.array(circuit.register_data["x"])
    scratch = km.array(
        [km.q(i) for i in range(circuit.num_qubits) if km.q(i) not in set(x)]
    )
    sim.write_across_shots(
        x, ((xs[None, :] >> np.arange(22, dtype=np.uint64)[:, None]) & 1).astype(bool)
    )
    sim.do(circuit)
    actual_x = (
        sim.read_across_shots(x).astype(np.uint64)
        << np.arange(22, dtype=np.uint64)[:, None]
    ).sum(axis=0)
    phase = np.array([float(sim.read_shot_phase(i) / 2) % 1 for i in range(len(xs))])
    difference = np.abs((phase - expected + 0.5) % 1 - 0.5)
    stats = sim.operation_statistics()
    report = dict(
        seed=seed,
        shots=len(xs),
        x_restored=bool(np.array_equal(actual_x, xs)),
        scratch_clean=bool(not sim.read_across_shots(scratch).any()),
        max_model_difference_turns=float(difference.max()),
        mean_toffoli_like=float(stats.num_toffoli_like / len(xs)),
        formal_verification=False,
        scope="Local native sample check, not the full official verifier.",
    )
    if (
        not report["x_restored"]
        or not report["scratch_clean"]
        or difference.max() > 1e-15
    ):
        raise AssertionError(json.dumps(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--address-bits", type=int, default=8)
    parser.add_argument("--fraction-bits", type=int, default=31)
    parser.add_argument("--slope-bits", type=int, default=25)
    parser.add_argument("--curvature-bits", type=int, default=15)
    parser.add_argument(
        "--arithmetic", choices=("explicit", "native"), default="explicit"
    )
    parser.add_argument("--max-qubits", type=int)
    args = parser.parse_args()
    config = QuadraticConfig(
        args.address_bits, args.fraction_bits, args.slope_bits, args.curvature_bits
    )
    table = fit_quadratic(config)
    numerical = screen(config, table)
    if numerical["status"] != "MODEL_PASS":
        raise ValueError("Candidate failed the full-domain numerical screen.")
    assembly, report = build_quadratic(
        config, table, arithmetic=args.arithmetic, max_qubits=args.max_qubits
    )
    circuit = km.Circuit.from_file(io.BytesIO(assembly.encode()), format="kmx")
    xs = np.r_[
        np.arange(1, 65),
        [65535, 65536, 65537, 1048576, 2097151, 2097152, 2097153, 3133452],
        np.random.default_rng(4).integers(1, 3133453, 184),
    ].astype(np.uint64)
    expected = quadratic_model(xs, config, table)
    samples = [check_native(circuit, xs, expected, seed=seed) for seed in (0, 1, 17)]
    report.update(
        numerical_screen=numerical,
        native_samples=samples,
        kickmix_path=km.__file__,
        baseline_cost=140719.08607029635,
        selection=(
            "REJECTED_COST_LOWER_BOUND"
            if report["cost_lower_bound"] >= 140719.08607029635
            else "REQUIRES_FURTHER_COST_AND_OFFICIAL_VERIFICATION"
        ),
    )
    buffer = io.BytesIO()
    circuit.to_file(buffer, format="kmb")
    report["circuit_sha256"] = hashlib.sha256(buffer.getvalue()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "invsqrt.kmx").write_bytes(assembly.encode())
    (args.output_dir / "invsqrt.kmb").write_bytes(buffer.getvalue())
    np.save(args.output_dir / "coefficients.npy", table, allow_pickle=False)
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
