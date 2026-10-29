"""Compare three exact chemistry ANGLE batchings; verify only a strict model improvement."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile


BATCHES = (1, 2, 3)
DEFAULT_SALT = "quantum-verification-workbench-20261025-v1"
PROJECTION_SCOPE = (
    "Local projection from the verified chemistry cost ratio only, assuming byte-identical "
    "invsqrt and dqi. Whole-submission fresh verification and Kaggle scoring are pending."
)


def validate_batch(value: int) -> int:
    if type(value) is not int or value not in BATCHES:
        raise ValueError("Only integer J=1,2,3 is allowed; J>=4 cannot fit the 128-qubit cap.")
    return value


def _positive_cost(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def select_candidate(trials: list[dict], baseline_cost: float) -> dict | None:
    """Select a model-pending trial, never declare it a verified winner."""
    if not _positive_cost(baseline_cost):
        raise ValueError("Baseline cost must be finite and positive.")
    eligible = [
        trial for trial in trials
        if trial.get("status") == "PENDING" and type(trial.get("J")) is int
        and trial["J"] in BATCHES and _positive_cost(trial.get("predicted_cost"))
        and trial["predicted_cost"] < baseline_cost
    ]
    return min(eligible, key=lambda trial: (trial["predicted_cost"], trial["J"])) if eligible else None


def accept_verified_candidate(trial: dict, verification: dict | None, baseline_cost: float, expected_shots: int) -> dict | None:
    """Require a fresh, full-strength chemistry-only report before declaring an improvement."""
    validate_batch(trial.get("J"))
    if trial.get("status") != "PENDING" or not isinstance(verification, dict):
        raise ValueError("A pending model trial needs a completed fresh verification report.")
    if select_candidate([trial], baseline_cost) is None:
        raise ValueError("Only a valid, strictly improving model trial can be promoted.")
    if (verification.get("status") != "PARTIAL_PASS" or verification.get("verification_mode") != "fresh_full"
        or verification.get("cached_results_used") is not False
        or verification.get("complete_submission_verification") is not False
        or set(verification.get("tasks", {})) != {"dfthcsos"}):
        raise ValueError("Expected a fresh chemistry-only PARTIAL_PASS, not an overall submission PASS.")
    result = verification["tasks"]["dfthcsos"]
    if (result.get("status") != "PASS" or result.get("fresh_verification") is not True
        or result.get("verified_shots") != expected_shots or not _positive_cost(result.get("cost"))):
        raise ValueError("Chemistry verification failed, used fewer shots, or returned an invalid cost.")
    if not _positive_cost(baseline_cost):
        raise ValueError("Baseline cost must be finite and positive.")
    cost = result["cost"]
    if cost >= baseline_cost:
        return None
    ratio = baseline_cost / cost
    return {
        "status": "CHEMISTRY_PASS", "J": trial["J"], "baseline_cost": baseline_cost,
        "candidate_cost": cost, "chemistry_improvement_ratio": ratio,
        "projected_total_score_multiplier": ratio ** (1 / 3), "projection_scope": PROJECTION_SCOPE,
        "whole_submission_fresh_verification": "PENDING", "verified_shots": expected_shots,
    }


def _digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _save(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=_json_value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("chemistry_sweep_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _package_candidate(public: Path, chemistry: Path, destination: Path) -> dict:
    members = {"invsqrt.kmb": public / "invsqrt.kmb", "dqi.kmb": public / "dqi.kmb"}
    members.update({"dfthcsos/" + path.name: path for path in chemistry.glob("angle_k_*.kmb")})
    members["dfthcsos/submission.npz"] = chemistry / "submission.npz"
    expected = {name: _digest(path) for name, path in members.items()}
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(members.items()):
            archive.write(path, name)
    with zipfile.ZipFile(destination) as archive:
        if set(archive.namelist()) != set(members):
            raise ValueError("Packaged submission member set changed.")
        for name, expected_digest in expected.items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected_digest:
                    raise ValueError("Packaged bytes differ: " + name)
    return {"path": str(destination), "sha256": _digest(destination), "members_sha256": expected}


def run_sweep(data_root: Path, public_submission: Path, output_dir: Path, tools_dir: Path, salt: str = DEFAULT_SALT) -> int:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Use a fresh output directory: " + str(output_dir))
    output_dir.mkdir(parents=True)
    summary = {"status": "RUNNING", "scope": "chemistry_only", "salt": salt, "batches": list(BATCHES),
               "trials": [], "winner": None, "whole_submission_fresh_verification": "PENDING"}
    _save(output_dir / "sweep.json", summary)
    try:
        if not isinstance(salt, str) or len(salt) < 32:
            raise ValueError("Local salt must contain at least 32 characters.")
        runner_path, metric_path = tools_dir / "verify_candidate.py", tools_dir / "official_metric.ipynb"
        runner = _load_runner(runner_path)
        backend = runner.load_backend(data_root)
        CV = backend.dfthcsos
        import numpy as np
        sys.path.insert(0, str(tools_dir))
        import learned_factor_compile as learned

        chemistry = public_submission / "dfthcsos"
        original_npz = chemistry / "submission.npz"
        original_npz_hash = _digest(original_npz)
        summary["upstream_members"] = [
            {"path": str(path.relative_to(public_submission)), "sha256": _digest(path), "bytes": path.stat().st_size}
            for path in sorted(public_submission.rglob("*")) if path.is_file()
        ]
        summary["source_sha256"] = {
            path.name: _digest(path) for path in
            (runner_path, metric_path, tools_dir / "learned_qrom.py", tools_dir / "learned_factor_compile.py", Path(__file__))
        }
        summary["official_backend_sha256"] = {key: _digest(Path(module.__file__)) for key, module in vars(backend).items()}
        print("Recovering baseline words with full 64-shots-per-address official evaluation.", flush=True)
        started = time.monotonic()
        with (output_dir / "baseline.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            inputs = CV.load_inputs(original_npz)
            edges = CV.load_tree_edges(None, chemistry)
            baseline = CV.evaluate_circuits(inputs, chemistry, edges, shots_per_address=64, salt=salt)
        reason = CV.rejection_reason(baseline["metrics"], baseline["resources"])
        words = np.array(baseline["angle_words"], dtype="<u2", copy=True)
        expected_shots = len(words) * 64
        baseline_cost = float(baseline["score"])
        if (reason is not None or baseline["brot"] != 10 or words.shape != (1636, 75)
            or baseline["verified_shots"] != expected_shots or not _positive_cost(baseline_cost)):
            raise ValueError("Baseline failed full chemistry validation: " + str(reason))
        baseline_report = {key: value for key, value in baseline.items() if key != "angle_words"}
        baseline_report.update(status="PASS", scope="chemistry_only", seconds=time.monotonic() - started,
                               shots_per_address=64, npz_sha256=original_npz_hash,
                               angle_words_sha256=hashlib.sha256(words.tobytes(order="C")).hexdigest())
        _save(output_dir / "baseline.json", baseline_report)
        summary["baseline"] = baseline_report
        with np.load(original_npz, allow_pickle=False) as archive:
            compact = {key: np.array(archive[key], copy=True) for key in learned.COMPACT_KEYS if key != "angle_words"}
        compact["angle_words"] = words
        learned.validate_compact(compact, brot=10)
        np.savez_compressed(output_dir / "fixed_compact.npz", **compact)
        del inputs, baseline
        gc.collect()

        for batch in BATCHES:
            validate_batch(batch)
            trial_dir = output_dir / f"J{batch}" / "dfthcsos"
            trial = {"J": batch, "status": "BUILDING", "directory": str(trial_dir), "predicted_cost": None}
            summary["trials"].append(trial)
            _save(output_dir / "sweep.json", summary)
            circuits = None
            try:
                print(f"Building exact J={batch} family, fixed words and original NPZ bytes.", flush=True)
                with (output_dir / f"J{batch}.build.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    circuits, _, build = learned.build_family(compact, trial_dir, brot=10, J=batch, canonicalize=False)
                shutil.copyfile(original_npz, trial_dir / "submission.npz")
                if _digest(trial_dir / "submission.npz") != original_npz_hash:
                    raise ValueError("Original factorization bytes changed.")
                resources = build["official_resource_model"]
                rejection = CV.rejection_reason(baseline_report["metrics"], resources)
                predicted = float(CV.score_from_metrics(baseline_report["metrics"], resources))
                if rejection is not None or not _positive_cost(predicted):
                    raise ValueError("Rejected resource model: " + str(rejection))
                trial.update(status="PENDING", predicted_cost=predicted, resources=resources,
                             npz_sha256=original_npz_hash, num_circuits=len(circuits),
                             expected_shots=expected_shots, full_official_verification="NOT_RUN")
                trial["angle_sha256"] = {path.name: _digest(path) for path in sorted(trial_dir.glob("angle_k_*.kmb"))}
            except Exception as error:
                trial.update(status="FAIL", error=f"{type(error).__name__}: {error}")
                with (output_dir / f"J{batch}.error.log").open("w", encoding="utf-8") as log:
                    traceback.print_exc(file=log)
            finally:
                del circuits
                gc.collect()
            _save(output_dir / "sweep.json", summary)

        if not any(trial["status"] == "PENDING" for trial in summary["trials"]):
            summary["failure_code"] = "NO_VALID_CANDIDATES"
            raise RuntimeError("NO_VALID_CANDIDATES: every batch build or resource model failed.")
        selected = select_candidate(summary["trials"], baseline_cost)
        if selected is None:
            summary["status"] = "NO_IMPROVEMENT"
            _save(output_dir / "sweep.json", summary)
            print("NO_IMPROVEMENT: no strictly better valid model candidate.", flush=True)
            return 0
        summary["selected_model_trial"] = selected["J"]
        _save(output_dir / "sweep.json", summary)
        verification_dir = output_dir / "winner_verification"
        command = [sys.executable, "-u", str(runner_path), "--data-root", str(data_root),
                   "--submission-dir", str(Path(selected["directory"]).parent), "--output-dir", str(verification_dir),
                   "--metric", str(metric_path), "--salt", salt, "--tasks", "dfthcsos"]
        print(f"Fresh full chemistry verification for model-selected J={selected['J']}.", flush=True)
        with (output_dir / "winner_verification.console.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=3 * 60 * 60, check=False)
        if process.returncode:
            raise RuntimeError("Selected candidate failed official chemistry verification.")
        verification = json.loads((verification_dir / "results.json").read_text(encoding="utf-8"))
        winner = accept_verified_candidate(selected, verification, baseline_cost, expected_shots)
        if winner is None:
            summary["status"] = "NO_IMPROVEMENT"
        else:
            if _digest(original_npz) != original_npz_hash or _digest(Path(selected["directory"]) / "submission.npz") != original_npz_hash:
                raise ValueError("Factorization bytes changed before packaging.")
            package = _package_candidate(public_submission, Path(selected["directory"]), output_dir / "submission.zip")
            winner.update(package=package, verification_results_sha256=_digest(verification_dir / "results.json"))
            summary.update(status="CHEMISTRY_IMPROVEMENT_VERIFIED", winner=winner)
        _save(output_dir / "sweep.json", summary)
        print(json.dumps({"status": summary["status"], "winner": summary["winner"]}, default=_json_value), flush=True)
        return 0
    except Exception as error:
        summary.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        with (output_dir / "error.log").open("w", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        _save(output_dir / "sweep.json", summary)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--public-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tools-dir", type=Path, required=True)
    parser.add_argument("--salt", default=DEFAULT_SALT)
    args = parser.parse_args(argv)
    return run_sweep(args.data_root, args.public_submission, args.output_dir, args.tools_dir, args.salt)


if __name__ == "__main__":
    raise SystemExit(main())
