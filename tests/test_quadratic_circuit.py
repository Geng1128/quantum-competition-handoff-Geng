"""Native circuit checks independent of the integer fitter/model."""

import importlib.util
import io
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_generator():
    folder = ROOT / "experiments/invsqrt"
    sys.path.insert(0, str(folder))
    try:
        import reconstruct

        script = folder / "quadratic_circuit.py"
        assert (
            script.exists()
        ), "The independent quadratic circuit generator is missing."
        spec = importlib.util.spec_from_file_location("quadratic_circuit", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return reconstruct, module
    finally:
        sys.path.remove(str(folder))


def test_generated_circuit_matches_model_and_cleans_scratch_for_multiple_seeds():
    km = pytest.importorskip("kickmix")
    model, generator = load_generator()
    config = model.QuadraticConfig()
    table = model.fit_quadratic(config)
    body, metadata = generator.build_quadratic(config, table)
    circuit = km.Circuit.from_file(io.BytesIO(body.encode()), format="kmx")
    xs = np.r_[
        np.arange(1, 65),
        [65535, 65536, 65537, 1048576, 2097151, 2097152, 2097153, 3133452],
        np.random.default_rng(4).integers(1, 3133453, 184),
    ].astype(np.uint64)
    assert len(xs) == 256
    expected = model.quadratic_model(xs, config, table)
    for seed in [0, 1, 17]:
        report = generator.check_native(circuit, xs, expected, seed=seed)
        assert report["x_restored"]
        assert report["scratch_clean"]
        assert report["max_model_difference_turns"] < 1e-15
        assert report["formal_verification"] is False
    assert metadata["formal_verification"] is False


def test_generator_rejects_out_of_range_table():
    pytest.importorskip("kickmix")
    model, generator = load_generator()
    config = model.QuadraticConfig()
    table = model.fit_quadratic(config)
    table[0, 1] = -1
    with pytest.raises(ValueError):
        generator.build_quadratic(config, table)


@pytest.mark.parametrize(
    "invalid", [np.full(256, np.nan), np.ones((128, 2)), np.ones(255)]
)
def test_native_check_rejects_invalid_expected_phase(invalid):
    km = pytest.importorskip("kickmix")
    _, generator = load_generator()
    declarations = (
        "".join(f"APPEND_TO_REGISTER q{i} r0\n" for i in range(22))
        + 'REGISTER r0 "x"\n'
    )
    empty = km.Circuit.from_file(io.BytesIO(declarations.encode()), format="kmx")
    with pytest.raises(ValueError, match="expected"):
        generator.check_native(empty, np.arange(1, 257), invalid)


@pytest.mark.parametrize(
    "invalid", [np.zeros(256, dtype=int), np.full(256, 3133453), np.full(256, 1.5)]
)
def test_native_check_rejects_invalid_input_domain(invalid):
    km = pytest.importorskip("kickmix")
    _, generator = load_generator()
    declarations = (
        "".join(f"APPEND_TO_REGISTER q{i} r0\n" for i in range(22))
        + 'REGISTER r0 "x"\n'
    )
    empty = km.Circuit.from_file(io.BytesIO(declarations.encode()), format="kmx")
    with pytest.raises(ValueError, match="xs"):
        generator.check_native(empty, invalid, np.zeros(256))


def test_native_controlled_rows_preserve_model_under_scratch_budget():
    km = pytest.importorskip("kickmix")
    model, generator = load_generator()
    config = model.QuadraticConfig(8, 28, 22, 12)
    table = model.fit_quadratic(config)
    body, metadata = generator.build_quadratic(
        config, table, arithmetic="native", max_qubits=120
    )
    assert metadata["num_qubits"] <= 120
    circuit = km.Circuit.from_file(io.BytesIO(body.encode()), format="kmx")
    xs = np.r_[
        np.arange(1, 65), np.random.default_rng(23).integers(1, 3133453, 192)
    ].astype(np.uint64)
    expected = model.quadratic_model(xs, config, table)
    for seed in (0, 17):
        result = generator.check_native(circuit, xs, expected, seed=seed)
        assert result["scratch_clean"] and result["x_restored"]


def test_qubit_budget_covers_normalization_template_too():
    pytest.importorskip("kickmix")
    model, generator = load_generator()
    config = model.QuadraticConfig(fraction_bits=24)
    with pytest.raises(ValueError, match="budget"):
        generator.build_quadratic(
            config,
            np.zeros((256, 3), dtype=np.int64),
            arithmetic="native",
            max_qubits=63,
        )
