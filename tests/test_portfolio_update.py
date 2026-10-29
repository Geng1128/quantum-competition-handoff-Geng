import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_portfolio_update", ROOT / "scripts/prepare_portfolio_update.py")
portfolio = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(portfolio)


@pytest.fixture
def evidence():
    blobs = portfolio.parent_artifacts(ROOT)
    return portfolio.decode_evidence(blobs, {key: hashlib.sha256(blob).hexdigest() for key, blob in blobs.items()})


def fresh_portfolio(parent, factor=.75):
    result = copy.deepcopy(parent["public"])
    task = result["tasks"]["dqi"]
    cost = task["cost"] * factor
    task.update(cost=cost, files=[{"path": "dqi.kmb", "sha256": portfolio.NEW_DQI_SHA256, "bytes": portfolio.NEW_DQI_BYTES}])
    task["raw"].update(score=cost, circuit_sha256=portfolio.NEW_DQI_SHA256)
    return result


def test_history_provides_only_cost_reference_not_current_pass(evidence):
    reference = portfolio.historical_reference(evidence)
    assert reference["usage"] == "historical_reference_costs"
    assert reference["old_local_score"] == evidence["comparison"]["local_score"]
    assert "current_backend_sha256" not in reference


@pytest.mark.parametrize("field,value", [("status", "FAIL"), ("complete_submission_verification", False), ("cached_results_used", True)])
def test_failed_or_partial_history_cannot_supply_cost_baseline(evidence, field, value):
    evidence["public"][field] = value
    with pytest.raises(ValueError):
        portfolio.historical_reference(evidence)


def test_current_backend_must_match_expected_manifest_independently():
    assert portfolio.check_current_backend({"kickmix.so": "a"}, {"kickmix.so": "a"}) is None
    with pytest.raises(ValueError):
        portfolio.check_current_backend({"kickmix.so": "a"}, {"kickmix.so": "b"})
    with pytest.raises(ValueError):
        portfolio.check_current_backend({}, {})


def test_duplicate_json_keys_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        portfolio.strict_json(b'{"tasks":{"dqi":{},"dqi":{}}}')


def test_all_three_fresh_passes_produce_measured_improvement(evidence):
    result = portfolio.combine_results(portfolio.historical_reference(evidence), fresh_portfolio(evidence))
    assert result["status"] == "IMPROVEMENT_VERIFIED_FRESH_FULL"
    assert result["verification_mode"] == "fresh_full_portfolio_update"
    assert result["whole_submission_fresh_this_run"] is True
    assert result["cached_results_used"] is False
    assert all(row["fresh_this_run"] is True for row in result["tasks"].values())
    assert result["new_local_score"] == pytest.approx(evidence["comparison"]["local_score"] * (1/.75)**(1/3))
    assert result["reference_cost_source"]["usage"] == "historical_reference_costs"
    assert portfolio.combine_results(portfolio.historical_reference(evidence), fresh_portfolio(evidence, 1.1))["status"] == "NO_IMPROVEMENT"


@pytest.mark.parametrize("field,value", [
    ("status", "FAIL"), ("status", "PARTIAL_PASS"), ("complete_submission_verification", False),
    ("verification_mode", "fresh_dqi_with_hash_exact_component_reuse"), ("cached_results_used", True),
])
def test_no_partial_or_historical_pass_can_claim_current_full_pass(evidence, field, value):
    fresh = fresh_portfolio(evidence)
    fresh[field] = value
    with pytest.raises(ValueError):
        portfolio.combine_results(portfolio.historical_reference(evidence), fresh)


@pytest.mark.parametrize("mutation", ["missing", "extra", "hash", "shots", "negative", "nan", "nonfresh", "salt"])
def test_current_all_task_gate_rejects_incomplete_or_invalid_evidence(evidence, mutation):
    fresh = fresh_portfolio(evidence)
    if mutation == "missing":
        del fresh["tasks"]["dfthcsos"]
    elif mutation == "extra":
        fresh["tasks"]["duplicate_dqi"] = copy.deepcopy(fresh["tasks"]["dqi"])
    elif mutation == "hash":
        fresh["tasks"]["dqi"]["files"][0]["sha256"] = "0"*64
    elif mutation == "shots":
        fresh["tasks"]["invsqrt"]["verified_shots"] = 64
    elif mutation == "negative":
        fresh["tasks"]["dqi"]["cost"] = -1
    elif mutation == "nan":
        fresh["tasks"]["dfthcsos"]["cost"] = float("nan")
    elif mutation == "nonfresh":
        fresh["tasks"]["dfthcsos"]["fresh_verification"] = False
    else:
        fresh["salt"] = "different salt"
    with pytest.raises(ValueError):
        portfolio.combine_results(portfolio.historical_reference(evidence), fresh)


def test_prepared_job_runs_all_three_tasks_and_does_not_modify_original_job(tmp_path):
    original = (ROOT / "kaggle/quantum-verification-workbench/quantum-verification-workbench.ipynb").read_bytes()
    prepared = portfolio.prepare_job(ROOT, tmp_path / "job", "gengsr")
    assert prepared["uploaded"] is False
    metadata = json.loads((tmp_path / "job/kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["is_private"] is True and metadata["enable_gpu"] is False and metadata["enable_internet"] is False
    assert metadata["kernel_sources"][-1] == portfolio.NEW_KERNEL
    notebook = json.loads((tmp_path / "job" / metadata["code_file"]).read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        compile("".join(cell["source"]), "generated", "exec")
    command_seen = []

    def failed_run(command, **kwargs):
        command_seen.append(command)
        return SimpleNamespace(returncode=1)
    namespace = {"RUNS": tmp_path, "WORK": tmp_path, "TOOLS": tmp_path,
                 "CANDIDATE": tmp_path, "DATA_ROOT": tmp_path, "sys": SimpleNamespace(executable="python"),
                 "SALT": portfolio.SALT, "TASKS": portfolio.TASKS,
                 "subprocess": SimpleNamespace(run=failed_run, STDOUT=-2)}
    with pytest.raises(RuntimeError, match="no portfolio"):
        exec("".join(notebook["cells"][-1]["source"]), namespace)
    assert command_seen[0][command_seen[0].index("--tasks")+1:] == list(portfolio.TASKS)
    assert not (tmp_path / "submission.zip").exists()
    assert original == (ROOT / "kaggle/quantum-verification-workbench/quantum-verification-workbench.ipynb").read_bytes()
