"""Independent native and integer checks for deterministic phase refitting."""

import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "experiments/invsqrt/phase_refit.py"
    assert path.exists(), "Phase refit implementation is missing."
    spec = importlib.util.spec_from_file_location("phase_refit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packed_trace_matches_full_native_baseline_phase():
    km = pytest.importorskip("kickmix")
    m = load_module()
    body = (ROOT / "experiments/invsqrt/baseline/invsqrt.kmx").read_bytes().decode()
    xs = np.r_[
        np.arange(1, 65), np.random.default_rng(7).integers(1, 3133453, 192)
    ].astype(np.uint64)
    features, sites = m.trace_features(body, xs)
    coefficients = np.array([s["phase_units"] for s in sites], dtype=np.uint64)
    phase = m.phase_units(features, coefficients)
    circuit = km.Circuit.from_file(io.BytesIO(body.encode()), format="kmx")
    for seed in (1, 17):
        sim = km.Simulator(batch_size=len(xs), seed=seed)
        x = km.array(circuit.register_data["x"])
        sim.write_across_shots(
            x,
            ((xs[None, :] >> np.arange(22, dtype=np.uint64)[:, None]) & 1).astype(bool),
        )
        sim.do(circuit)
        actual = np.array(
            [m.half_turns_to_units(sim.read_shot_phase(i)) for i in range(len(xs))],
            dtype=np.uint64,
        )
        difference = (phase - actual).view(np.int64)
        assert np.abs(difference).max() <= len(sites)


def test_reference_units_known_values():
    m = load_module()
    target, tolerance = m.reference_units(np.array([1, 4], dtype=np.uint64))
    assert target.tolist() == [0, 1 << 63]
    assert tolerance.tolist() == [1 << 40, 1 << 39]


def test_cutting_plane_solver_corrects_all_basis_patterns():
    m = load_module()
    h = 1 << 20
    features = np.array([[0, 1, 0, 1], [0, 0, 1, 1]], dtype=np.bool_)
    base = np.array([20 * h, 40 * h], dtype=np.uint64)
    target = np.array([0, 30 * h, 50 * h, 80 * h], dtype=np.uint64)
    tolerance = np.full(4, 16 * h, dtype=np.uint64)
    result = m.solve_refit(features, base, target, tolerance, [0, 1])
    assert result["status"] == "MODEL_FULL_DOMAIN_PASS"
    new = np.array(result["coefficients_units"], dtype=np.uint64)
    np.testing.assert_array_equal(m.phase_units(features, new), target)
    impossible = m.solve_refit(features, base, target, tolerance, [0, 1], dropped=[0])
    assert impossible["status"] == "INFEASIBLE"


def test_trace_refuses_classical_feedback_that_changes_bits():
    m = load_module()
    with pytest.raises(ValueError):
        m.trace_features("X q0 if b0\nZ_POW q0 0.25\n", np.array([1]))


def test_dual_witness_identifies_missing_parity_feature():
    m = load_module()
    h = 1 << 20
    features = np.array([[0, 1, 0, 1], [0, 0, 1, 1]], dtype=np.bool_)
    target = np.array([0, 0, 0, 30 * h], dtype=np.uint64)
    tolerance = np.full(4, 20 * h, dtype=np.uint64)
    result = m.solve_refit(
        features, np.zeros(2, dtype=np.uint64), target, tolerance, [0, 1]
    )
    witness = result["dual_witness"]
    parity = features[0] ^ features[1]
    price = sum(row["weight"] * int(parity[row["input_index"]]) for row in witness)
    assert abs(price) > 0.1
    for column in features:
        assert (
            abs(sum(row["weight"] * int(column[row["input_index"]]) for row in witness))
            < 1e-8
        )
    expanded = np.vstack([features, parity])
    better = m.solve_refit(
        expanded, np.zeros(3, dtype=np.uint64), target, tolerance, [0, 1, 2]
    )
    assert better["min_margin_units"] > result["min_margin_units"]


@pytest.mark.parametrize("xs", [np.array([1.5]), np.array([True])])
def test_reference_rejects_noninteger_domain(xs):
    m = load_module()
    with pytest.raises(ValueError):
        m.reference_units(xs)


def test_solver_accepts_equivalent_nonnegative_signed_units():
    m = load_module()
    h = 1 << 20
    features = np.array([[0, 1, 0, 1], [0, 0, 1, 1]], dtype=np.bool_)
    base = np.array([20 * h, 40 * h], dtype=np.int64)
    target = np.array([0, 30 * h, 50 * h, 80 * h], dtype=np.int64)
    tolerance = np.full(4, 16 * h, dtype=np.int64)
    result = m.solve_refit(features, base, target, tolerance, [0, 1])
    assert result["status"] == "MODEL_FULL_DOMAIN_PASS"


def test_rewrite_rejects_truncated_coefficients():
    m = load_module()
    with pytest.raises(ValueError):
        m.rewrite_phases("Z_POW q0 0.25\n", [{"line": 1, "wire": 0}], [])


def test_rewrite_rejects_duplicate_sites():
    m = load_module()
    with pytest.raises(ValueError):
        m.rewrite_phases("Z_POW q0 0.25\n", [{"line": 1, "wire": 0}] * 2, [1, 2])
