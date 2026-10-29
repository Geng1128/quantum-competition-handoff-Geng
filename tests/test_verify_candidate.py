import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_candidate.py"
spec = importlib.util.spec_from_file_location("verify_candidate", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def submission(tmp_path):
    root = tmp_path / "candidate"
    (root / "dfthcsos").mkdir(parents=True)
    for name in ("invsqrt.kmb", "dqi.kmb", "dfthcsos/angle_k_0.kmb", "dfthcsos/angle_k_1.kmb", "dfthcsos/submission.npz"):
        (root / name).write_bytes(b"test fixture only")
    return root


@pytest.fixture
def backend(monkeypatch):
    calls = []

    def verify(name):
        def invoke(*args, **kwargs):
            calls.append((name, args, kwargs))
            return {"status": "PASS", "score": 123.5, "num_qubits": 22, "verified_shots": 102400}
        return invoke

    metric = SimpleNamespace(
        read_circuit=lambda path, cap: SimpleNamespace(num_qubits=22),
        read_npz=lambda path, spec: {"tree_edges": [[0, 1]], "U": "test arrays"},
    )
    modules = SimpleNamespace(
        invsqrt=SimpleNamespace(verify=verify("invsqrt"), ProblemSpecInvSqrt=SimpleNamespace(for_n_p=lambda n: ("invsqrt_spec", n))),
        dqi=SimpleNamespace(verify=verify("dqi"), ProblemSpecDQI=SimpleNamespace(benchmark_instance=lambda: SimpleNamespace(n=70, m=4095))),
        dfthcsos=SimpleNamespace(verify=verify("dfthcsos"), QUBIT_CAP=128, SHOTS_PER_ADDRESS=64),
        reference=SimpleNamespace(ORBITALS=76, FACTORIZATION_KEYS=("U", "W", "H2_I", "H1_I", "Hsym_lower")),
        kickmix=SimpleNamespace(__file__="test backend; no simulation"),
    )
    monkeypatch.setattr(runner, "load_metric", lambda path: metric)
    monkeypatch.setattr(runner, "load_backend", lambda data_root: modules)
    return SimpleNamespace(calls=calls, metric=metric, modules=modules)


def execute(submission, tmp_path, **kwargs):
    return runner.run_verification(
        data_root=tmp_path / "official", submission_dir=submission,
        output_dir=tmp_path / "results", metric_path=Path(__file__),
        salt="test-salt-for-full-local-verification-20261025", **kwargs,
    )


@pytest.mark.parametrize("filename", ["invsqrt.kmb", "dqi.kmb", "dfthcsos/submission.npz"])
def test_missing_required_file_fails_before_verification(submission, tmp_path, backend, filename):
    (submission / filename).unlink()
    assert execute(submission, tmp_path) == 1
    result = json.loads((tmp_path / "results/results.json").read_text(encoding="utf-8"))
    assert result["status"] == "FAIL"
    assert backend.calls == []


def test_duplicate_circuit_formats_rejected(submission):
    (submission / "invsqrt.kmx").write_text("not a circuit")
    with pytest.raises(ValueError, match="exactly one"):
        runner.collect_submission_files(submission)


@pytest.mark.parametrize("extra", ["angle_k_0.kmx", "angle_k_3.kmb", "angle_k_00.kmb"])
def test_angle_duplicates_or_gaps_rejected(submission, extra):
    (submission / "dfthcsos" / extra).write_bytes(b"invalid family")
    with pytest.raises(ValueError, match="angle_k_0"):
        runner.collect_submission_files(submission)


def test_angle_files_use_integer_order(submission):
    for i in range(2, 12):
        (submission / "dfthcsos" / f"angle_k_{i}.kmb").write_bytes(b"test")
    files = runner.collect_submission_files(submission)["dfthcsos"]
    assert [path.name for path in files[:-1]] == [f"angle_k_{i}.kmb" for i in range(12)]


def test_full_run_records_fresh_results_and_official_defaults(submission, tmp_path, backend):
    assert execute(submission, tmp_path) == 0
    result = json.loads((tmp_path / "results/results.json").read_text(encoding="utf-8"))
    assert result["status"] == "PASS"
    assert result["complete_submission_verification"] is True
    assert result["verification_mode"] == "fresh_full"
    assert set(result["tasks"]) == set(runner.TASKS)
    inv = json.loads((tmp_path / "results/invsqrt.json").read_text(encoding="utf-8"))
    assert inv["status"] == "PASS" and inv["cost"] == 123.5
    assert inv["verified_shots"] == 102400 and inv["num_qubits"] == 22
    assert len(inv["files"][0]["sha256"]) == 64
    assert (tmp_path / "results/invsqrt.log").is_file()
    assert backend.calls[0][1][1] == ("invsqrt_spec", 10)
    assert backend.calls[2][2]["shots_per_address"] == 64
    assert all(call[2]["salt"].startswith("test-salt") for call in backend.calls)


def test_selected_task_cannot_claim_full_submission_pass(submission, tmp_path, backend):
    assert execute(submission, tmp_path, tasks=("invsqrt",)) == 0
    result = json.loads((tmp_path / "results/results.json").read_text(encoding="utf-8"))
    assert result["status"] == "PARTIAL_PASS"
    assert result["complete_submission_verification"] is False
    assert [call[0] for call in backend.calls] == ["invsqrt"]


def test_preflight_all_tasks_before_any_simulation(submission, tmp_path, backend):
    def invalid_npz(*args):
        raise ValueError("invalid chemistry array shape")
    backend.metric.read_npz = invalid_npz
    assert execute(submission, tmp_path) == 1
    assert backend.calls == []
    result = json.loads((tmp_path / "results/results.json").read_text(encoding="utf-8"))
    assert result["status"] == "FAIL"
    assert "invalid chemistry array shape" in result["error"]


@pytest.mark.parametrize("failure", [ValueError("bad phase"), RuntimeError("simulation failed")])
def test_verifier_exception_never_marked_pass(submission, tmp_path, backend, failure):
    def fail(*args, **kwargs):
        raise failure
    backend.modules.dqi.verify = fail
    assert execute(submission, tmp_path) == 1
    result = json.loads((tmp_path / "results/results.json").read_text(encoding="utf-8"))
    assert result["status"] == "FAIL"
    assert result["tasks"]["dqi"]["status"] == "FAIL"
    assert "Traceback" in (tmp_path / "results/dqi.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("bad_result", [
    {"status": "FAIL", "score": 2}, {"status": "PASS", "score": math.inf},
    {"status": "PASS", "score": 0}, {"status": "PASS", "score": math.nan},
])
def test_nonpassing_or_nonfinite_result_rejected(submission, tmp_path, backend, bad_result):
    backend.modules.invsqrt.verify = lambda *args, **kwargs: bad_result
    assert execute(submission, tmp_path) == 1
    assert json.loads((tmp_path / "results/results.json").read_text(encoding="utf-8"))["status"] == "FAIL"


@pytest.mark.parametrize("tasks", [(), ("unknown",), ("dqi", "dqi")])
def test_invalid_task_selection_rejected(submission, tasks):
    with pytest.raises(ValueError):
        runner.collect_submission_files(submission, tasks)


def test_short_salt_rejected(submission, tmp_path, backend):
    assert runner.run_verification(
        tmp_path / "official", submission, tmp_path / "out", "short", Path(__file__)
    ) == 1
    assert backend.calls == []


def test_cli_failure_returns_nonzero(tmp_path):
    assert runner.main([
        "--data-root", str(tmp_path), "--submission-dir", str(tmp_path / "missing"),
        "--output-dir", str(tmp_path / "out"),
    ]) == 1


@pytest.mark.parametrize("notebook", [False, True])
def test_load_metric_accepts_python_and_notebook(tmp_path, notebook):
    source = "def read_circuit(*args): return 'reader'\ndef read_npz(*args): return {}\n"
    path = tmp_path / ("metric.ipynb" if notebook else "metric.py")
    path.write_text(json.dumps({"cells": [
        {"cell_type": "markdown", "source": ["ignored"]},
        {"cell_type": "code", "source": source.splitlines(keepends=True)},
    ]}) if notebook else source, encoding="utf-8")
    metric = runner.load_metric(path)
    assert metric.read_circuit(None) == "reader"
    assert metric.read_npz(None) == {}


def test_load_metric_rejects_missing_official_readers(tmp_path):
    path = tmp_path / "bad_metric.py"
    path.write_text("read_circuit = None\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing read_circuit"):
        runner.load_metric(path)


def test_backend_must_come_from_official_data(tmp_path, monkeypatch):
    data = tmp_path / "official"
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: SimpleNamespace(__file__=str(data / f"{name}.py")))
    backend = runner.load_backend(data)
    assert Path(backend.kickmix.__file__).is_relative_to(data)
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: SimpleNamespace(__file__=str(tmp_path / "wrong.py")))
    with pytest.raises(RuntimeError, match="outside official data"):
        runner.load_backend(data)


def test_numpy_json_and_unsupported_objects(tmp_path):
    import numpy as np
    path = tmp_path / "numpy.json"
    runner._write_json(path, {"values": np.array([1, 2]), "scalar": np.int64(3)})
    assert json.loads(path.read_text(encoding="utf-8")) == {"values": [1, 2], "scalar": 3}
    with pytest.raises(TypeError, match="Not JSON serializable"):
        runner._write_json(path, object())


def test_updated_official_schema_requires_review(submission, tmp_path, backend):
    backend.modules.reference.FACTORIZATION_KEYS = ("changed_key",)
    assert execute(submission, tmp_path) == 1
    assert backend.calls == []
    result = json.loads((tmp_path / "results/results.json").read_text(encoding="utf-8"))
    assert "factorization keys changed" in result["error"]


def test_hardlinked_submission_file_rejected(submission):
    import os
    os.link(submission / "invsqrt.kmb", submission / "linked_copy.bin")
    with pytest.raises(ValueError, match="not a link"):
        runner.collect_submission_files(submission)
