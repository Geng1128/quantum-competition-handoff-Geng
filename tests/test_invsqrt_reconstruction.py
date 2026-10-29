"""Independent checks for the reconstructed lookup and numerical search model."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/invsqrt/reconstruct.py"


def load_module():
    assert SCRIPT.exists(), "The invsqrt reconstruction module is not implemented yet."
    spec = importlib.util.spec_from_file_location("invsqrt_reconstruct", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_anf_lookup_recovers_every_output_word():
    module = load_module()
    table = [19, 2, 7, 31, 11, 0, 9, 23]
    coefficients = module.anf_coefficients(table)
    for address, expected in enumerate(table):
        actual = 0
        for mask, word in enumerate(coefficients):
            if address & mask == mask:
                actual ^= word
        assert actual == expected


def test_baseline_lookup_reconstruction_is_byte_identical():
    module = load_module()
    original = (ROOT / "experiments/invsqrt/baseline/invsqrt.kmx").read_bytes().decode()
    table = module.recover_lookup(original)
    replay = module.replay_baseline(original, table)
    assert replay == original
    assert len(table) == 256


@pytest.mark.parametrize("xs", [[0], [-1], [3133453], [1.5], [True]])
def test_normalization_rejects_invalid_inputs(xs):
    module = load_module()
    with pytest.raises(ValueError):
        module.normalize_inputs(np.asarray(xs))


def test_normalization_matches_exact_integer_identity():
    module = load_module()
    xs = np.array([1, 2, 3, 4, 15, 16, 65535, 65536, 2097152, 3133452])
    normalized, shifts, parity = module.normalize_inputs(xs)
    assert ((1 << 21) <= normalized).all()
    assert (normalized < (1 << 22)).all()
    np.testing.assert_array_equal(normalized, xs << (2 * shifts + parity))
    np.testing.assert_allclose(
        (2.0**shifts)
        * np.where(parity, np.sqrt(2.0), 1.0)
        / np.sqrt(normalized / 2.0**20),
        2.0**10 / np.sqrt(xs),
        rtol=1e-15,
    )


def test_quadratic_model_has_full_domain_margin():
    module = load_module()
    config = module.QuadraticConfig(
        address_bits=8, fraction_bits=31, slope_bits=25, curvature_bits=15
    )
    table = module.fit_quadratic(config)
    # Include all segment boundaries and several interior values, independently
    # of the fitting grid. The search CLI also checks all 3,133,452 inputs.
    xs = np.unique(
        np.r_[
            np.arange(1, 2000),
            np.arange(1 << 20, 3133453, 16384),
            np.arange(1 << 20, 3133453, 16384) + 8191,
            3133452,
        ]
    )
    xs = xs[xs <= 3133452]
    phase = module.quadratic_model(xs, config, table)
    error = np.abs(phase * np.sqrt(xs) - 1)
    assert error.max() < 2.0**-24


def test_table_shape_and_overflow_are_validated():
    module = load_module()
    with pytest.raises(ValueError):
        module.anf_coefficients([0, 1, 2])
    with pytest.raises(ValueError):
        module.QuadraticConfig(address_bits=1)


def test_replay_refuses_changed_coefficients_without_new_uncomputation():
    module = load_module()
    original = (ROOT / "experiments/invsqrt/baseline/invsqrt.kmx").read_bytes().decode()
    table = module.recover_lookup(original)
    table[0] ^= 1
    with pytest.raises(ValueError, match="uncomputation"):
        module.replay_baseline(original, table)


def test_unsigned_coefficient_table_matches_signed_model():
    module = load_module()
    config = module.QuadraticConfig()
    table = module.fit_quadratic(config)
    xs = np.array([1, 17, 65535, 3133452])
    np.testing.assert_array_equal(
        module.quadratic_model(xs, config, table.astype(np.uint64)),
        module.quadratic_model(xs, config, table),
    )


def test_model_preserves_multidimensional_input_shape_and_values():
    module = load_module()
    config = module.QuadraticConfig()
    table = module.fit_quadratic(config)
    xs = np.array([[1, 17], [65535, 3133452]])
    expected = module.quadratic_model(xs.ravel(), config, table).reshape(xs.shape)
    np.testing.assert_array_equal(module.quadratic_model(xs, config, table), expected)
