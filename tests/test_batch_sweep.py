import importlib.util
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("batch_sweep", ROOT / "experiments/chemistry/batch_sweep.py")
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


def pending(j, cost, status="PENDING"):
    return {"J": j, "predicted_cost": cost, "status": status}


@pytest.mark.parametrize("j", [0, 4, -1, 2.0, True, "2"])
def test_only_three_legal_integer_batches(j):
    with pytest.raises(ValueError):
        sweep.validate_batch(j)


def test_legal_domain_is_exactly_one_two_three():
    assert sweep.BATCHES == (1, 2, 3)
    assert [sweep.validate_batch(j) for j in sweep.BATCHES] == [1, 2, 3]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1, 0, True])
def test_invalid_model_costs_never_selected(bad):
    assert sweep.select_candidate([pending(1, bad)], 100) is None


def test_failed_and_non_strict_improvements_never_selected():
    trials = [pending(1, 90, "FAIL"), pending(2, 100), pending(3, 101)]
    assert sweep.select_candidate(trials, 100) is None


def test_only_best_pending_trial_selected():
    trials = [pending(1, 90), pending(2, 80), pending(3, 95)]
    assert sweep.select_candidate(trials, 100) == trials[1]
    assert all(trial["status"] == "PENDING" for trial in trials)


def verified_result(cost=80):
    return {
        "status": "PARTIAL_PASS", "verification_mode": "fresh_full",
        "cached_results_used": False, "complete_submission_verification": False,
        "tasks": {"dfthcsos": {"status": "PASS", "fresh_verification": True,
                                  "cost": cost, "verified_shots": 104704}},
    }


@pytest.mark.parametrize("field,value", [
    ("status", "RUNNING"), ("status", "PASS"),
    ("verification_mode", "fast"), ("cached_results_used", True),
    ("complete_submission_verification", True),
])
def test_unverified_or_mislabelled_candidate_cannot_be_winner(field, value):
    result = verified_result()
    result[field] = value
    with pytest.raises(ValueError):
        sweep.accept_verified_candidate(pending(2, 80), result, 100, 104704)


@pytest.mark.parametrize("field,value", [
    ("status", "FAIL"), ("fresh_verification", False),
    ("verified_shots", 1636), ("cost", math.nan), ("cost", 0),
])
def test_failed_or_short_verification_cannot_be_winner(field, value):
    result = verified_result()
    result["tasks"]["dfthcsos"][field] = value
    with pytest.raises(ValueError):
        sweep.accept_verified_candidate(pending(2, 80), result, 100, 104704)


def test_pending_model_trial_alone_is_not_a_verified_winner():
    with pytest.raises(ValueError):
        sweep.accept_verified_candidate(pending(2, 80), None, 100, 104704)


def test_invalid_model_trial_cannot_be_promoted_even_with_a_pass_report():
    with pytest.raises(ValueError):
        sweep.accept_verified_candidate(pending(2, math.nan), verified_result(), 100, 104704)


def test_verified_strict_improvement_projects_only_chemistry_gain():
    winner = sweep.accept_verified_candidate(pending(2, 70), verified_result(80), 100, 104704)
    assert winner["status"] == "CHEMISTRY_PASS"
    assert winner["chemistry_improvement_ratio"] == 1.25
    assert winner["projected_total_score_multiplier"] == pytest.approx(1.25 ** (1 / 3))
    assert winner["whole_submission_fresh_verification"] == "PENDING"
    assert sweep.accept_verified_candidate(pending(2, 70), verified_result(100), 100, 104704) is None


def test_builder_is_private_cpu_offline_and_preserves_compiler_source(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("prepare_chemistry_sweep", ROOT / "scripts/prepare_chemistry_sweep.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    output = tmp_path / "job"
    assert builder.main(["--project-root", str(ROOT), "--output-dir", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["uploaded"] is False
    metadata = json.loads((output / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "gengsr/quantum-chemistry-batch-sweep"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is False and metadata["enable_internet"] is False
    assert metadata["kernel_sources"] == ["anazemcev/quantum-circuits-joint-norm-and-compact-dqi/3"]
    notebook = json.loads((output / metadata["code_file"]).read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        compile("".join(cell["source"]), "generated_cell", "exec")
    sources = builder.compiler_sources(ROOT / "external/notebooks/public/quantum-circuits-joint-norm-and-compact-dqi.ipynb")
    assert list(sources) == ["learned_qrom.py", "learned_factor_compile.py"]
    assert b"SPDX-License-Identifier: Apache-2.0" in sources["learned_qrom.py"]
    assert b"canonicalize=False" in (ROOT / "experiments/chemistry/batch_sweep.py").read_bytes()


@pytest.mark.parametrize("predictions,expected_status", [
    ({1: 110, 2: 100, 3: 101}, "NO_IMPROVEMENT"),
    ({1: 90, 2: 80, 3: 85}, "CHEMISTRY_IMPROVEMENT_VERIFIED"),
    ({1: None, 2: None, 3: None}, "FAIL"),
])
def test_mock_pipeline_preserves_factor_bytes_and_requires_fresh_winner(tmp_path, monkeypatch, predictions, expected_status):
    import sys
    import types
    import zipfile
    import numpy as np
    public = tmp_path / "public"
    chemistry = public / "dfthcsos"
    chemistry.mkdir(parents=True)
    (public / "invsqrt.kmb").write_bytes(b"original invsqrt bytes")
    (public / "dqi.kmb").write_bytes(b"original dqi bytes")
    (chemistry / "angle_k_0.kmb").write_bytes(b"original angle bytes")
    np.savez(chemistry / "submission.npz", W=np.ones((1, 1, 1)), H2_I=np.ones((1, 1)),
             H1_I=np.array(0.), Hsym_lower=np.ones(3), tree_edges=np.array([[0, 1]]))
    original_npz = (chemistry / "submission.npz").read_bytes()
    tools = tmp_path / "tools"
    tools.mkdir()
    for filename in ("verify_candidate.py", "official_metric.ipynb", "learned_qrom.py", "learned_factor_compile.py"):
        (tools / filename).write_bytes(b"fixture source")
    cv = types.SimpleNamespace(
        __file__=str(tools / "official_metric.ipynb"),
        load_inputs=lambda path: "official inputs", load_tree_edges=lambda *args: np.array([[0, 1]]),
        evaluate_circuits=lambda *args, **kwargs: {
            "angle_words": np.zeros((1636, 75), dtype="<u2"), "brot": 10,
            "metrics": {"lambda_eff": 1.}, "resources": {"cost": 100},
            "score": 100., "verified_shots": 104704,
        },
        rejection_reason=lambda *args: None, score_from_metrics=lambda metrics, resources: resources["cost"],
    )
    monkeypatch.setattr(sweep, "_load_runner", lambda path: types.SimpleNamespace(load_backend=lambda data: types.SimpleNamespace(dfthcsos=cv)))
    built = []
    learned = types.ModuleType("learned_factor_compile")
    learned.COMPACT_KEYS = ("angle_words", "W", "H2_I", "H1_I", "Hsym_lower", "tree_edges")
    learned.validate_compact = lambda *args, **kwargs: None

    def build(compact, output, *, brot, J, canonicalize):
        assert brot == 10 and canonicalize is False
        built.append(J)
        if predictions[J] is None:
            raise RuntimeError("Fixture compiler build failure")
        output.mkdir(parents=True)
        (output / "angle_k_0.kmb").write_bytes(f"compiled J={J}".encode())
        (output / "submission.npz").write_bytes(b"rebuilt bytes must be overwritten")
        return (object(),), compact, {"official_resource_model": {"cost": predictions[J]}}
    learned.build_family = build
    monkeypatch.setitem(sys.modules, "learned_factor_compile", learned)
    monkeypatch.setattr(sweep.sys, "path", list(sys.path))
    verifications = []

    def verify(command, **kwargs):
        verifications.append(command)
        assert command[-2:] == ["--tasks", "dfthcsos"]
        destination = Path(command[command.index("--output-dir") + 1])
        destination.mkdir()
        (destination / "results.json").write_text(json.dumps(verified_result(80)), encoding="utf-8")
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(sweep.subprocess, "run", verify)
    output = tmp_path / "experiment"
    command = ["--data-root", str(tmp_path / "official"), "--public-submission", str(public),
               "--output-dir", str(output), "--tools-dir", str(tools)]
    if expected_status == "FAIL":
        with pytest.raises(RuntimeError, match="NO_VALID_CANDIDATES"):
            sweep.main(command)
    else:
        assert sweep.main(command) == 0
    summary = json.loads((output / "sweep.json").read_text(encoding="utf-8"))
    assert summary["status"] == expected_status
    assert built == [1, 2, 3]
    if expected_status == "FAIL":
        assert all(trial["status"] == "FAIL" for trial in summary["trials"])
        assert "NO_VALID_CANDIDATES" in summary["error"]
        assert summary["failure_code"] == "NO_VALID_CANDIDATES"
        assert summary["winner"] is None and verifications == []
        assert not (output / "submission.zip").exists()
        return
    assert all(trial["status"] == "PENDING" for trial in summary["trials"])
    for j in (1, 2, 3):
        assert (output / f"J{j}/dfthcsos/submission.npz").read_bytes() == original_npz
    if expected_status == "NO_IMPROVEMENT":
        assert summary["winner"] is None and verifications == []
        assert not (output / "submission.zip").exists()
    else:
        assert summary["winner"]["J"] == 2 and len(verifications) == 1
        with zipfile.ZipFile(output / "submission.zip") as archive:
            assert archive.read("invsqrt.kmb") == b"original invsqrt bytes"
            assert archive.read("dqi.kmb") == b"original dqi bytes"
            assert archive.read("dfthcsos/submission.npz") == original_npz


def test_unreviewed_compiler_notebook_is_rejected(tmp_path):
    spec = importlib.util.spec_from_file_location("prepare_chemistry_sweep", ROOT / "scripts/prepare_chemistry_sweep.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    path = tmp_path / "modified.ipynb"
    path.write_text('{"cells": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="source hash"):
        builder.compiler_sources(path)
