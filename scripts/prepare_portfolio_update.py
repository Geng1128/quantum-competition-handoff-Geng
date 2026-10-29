"""Prepare an all-three fresh full verification for a pinned circuit portfolio."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re


TASKS = ("invsqrt", "dqi", "dfthcsos")
SLUG = "quantum-portfolio-update"
SALT = "quantum-verification-workbench-20261025-v1"
NEW_KERNEL = "phucthaiv02/qcc-verified-portfolio-df-thc-retraining-study/9"
NEW_ARCHIVE_SHA256 = "99d26ffeca2ac5c0eb30795084d69b1a8227fb4a64ee2de157d4ef7a42dead48"
NEW_DQI_SHA256 = "c5da9cd4e2d73dd41417e539f2659009d646b3ee30ddb5ca494261378a502856"
NEW_DQI_BYTES = 869001577
OLD_ARCHIVE_SHA256 = "ede2fb264c13bd663dcc945a60f005a8008752d98a4483a979ae5287fc972e2f"
MODE = "fresh_full_portfolio_update"


def strict_json(data: bytes | str):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique)


def parent_artifacts(root: Path) -> dict[str, bytes]:
    parent = Path(root) / "runs/reproduction_v1"
    paths = {"comparison": parent / "comparison.json", "official": parent / "runs/official_baseline/results.json",
             "public": parent / "runs/public_baseline/results.json", "environment": parent / "runs/environment.json",
             "inputs": parent / "runs/input_manifest.json"}
    return {key: path.read_bytes() for key, path in paths.items()}


def decode_evidence(blobs: dict[str, bytes], expected_hashes: dict[str, str]) -> dict:
    if set(blobs) != {"comparison", "official", "public", "environment", "inputs"} or set(blobs) != set(expected_hashes):
        raise ValueError("Incomplete historical reference bundle.")
    for key, blob in blobs.items():
        if hashlib.sha256(blob).hexdigest() != expected_hashes[key]:
            raise ValueError("Historical artifact hash mismatch: " + key)
    result = {key: strict_json(blob) for key, blob in blobs.items()}
    result["_sha256"] = dict(expected_hashes)
    return result


def _positive(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _records(records: list[dict]) -> dict[str, tuple[str, int]]:
    result = {}
    for record in records:
        name = record.get("path")
        if (not isinstance(name, str) or "\\" in name or ":" in name
            or any(part in ("", ".", "..") for part in name.split("/"))):
            raise ValueError("Noncanonical component path.")
        if name in result:
            raise ValueError("Duplicate component path: " + name)
        if not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")) or type(record.get("bytes")) is not int or record["bytes"] <= 0:
            raise ValueError("Invalid component digest or size: " + name)
        result[name] = (record["sha256"], record["bytes"])
    return result


def _check_full_result(report: dict) -> None:
    if (report.get("status") != "PASS" or report.get("verification_mode") != "fresh_full"
        or report.get("complete_submission_verification") is not True or report.get("cached_results_used") is not False
        or set(report.get("tasks", {})) != set(TASKS) or report.get("salt") != SALT):
        raise ValueError("Expected an all-three fresh full PASS with the specified local salt.")
    for task in TASKS:
        row = report["tasks"][task]
        raw, shots = row.get("raw", {}), row.get("verified_shots")
        if (row.get("status") != "PASS" or row.get("fresh_verification") is not True or not _positive(row.get("cost"))
            or raw.get("status") != "PASS" or raw.get("score") != row["cost"]
            or type(shots) is not int or shots <= 0 or raw.get("verified_shots") != shots
            or type(row.get("num_qubits")) is not int or row["num_qubits"] <= 0 or raw.get("num_qubits") != row["num_qubits"]):
            raise ValueError("Invalid full task PASS evidence: " + task)
        if task in {"invsqrt", "dqi"} and shots != {"invsqrt": 200540928, "dqi": 102400}[task]:
            raise ValueError("Wrong full-strength shot count: " + task)
        _records(row.get("files", []))


def historical_reference(parent: dict) -> dict:
    """Validate measured historical costs; never treat history as this run's verification."""
    comparison = parent["comparison"]
    for field, artifact in (("official_results_sha256", "official"), ("public_results_sha256", "public"),
                            ("environment_sha256", "environment"), ("input_manifest_sha256", "inputs")):
        if comparison.get(field) != parent["_sha256"][artifact]:
            raise ValueError("Historical comparison artifact binding mismatch: " + artifact)
    if (comparison.get("status") != "PASS" or comparison.get("verification_mode") != "fresh_full"
        or set(comparison.get("tasks", {})) != set(TASKS) or comparison.get("local_salt") != SALT
        or comparison.get("public_archive_sha256") != OLD_ARCHIVE_SHA256
        or parent["inputs"].get("public_archive_sha256") != OLD_ARCHIVE_SHA256):
        raise ValueError("Historical comparison is not the expected measured reference.")
    for which in ("official", "public"):
        _check_full_result(parent[which])
        for task in TASKS:
            if comparison["tasks"][task].get(which + "_cost") != parent[which]["tasks"][task]["cost"]:
                raise ValueError("Historical task cost differs from comparison: " + task)
    expected = _records(parent["inputs"]["public_submission"])
    if _records([record for task in TASKS for record in parent["public"]["tasks"][task]["files"]]) != expected:
        raise ValueError("Historical archive file list does not match the measured portfolio.")
    old_score = math.exp(sum(math.log(parent["official"]["tasks"][task]["cost"] / parent["public"]["tasks"][task]["cost"]) for task in TASKS) / 3)
    if not _positive(comparison.get("local_score")) or not math.isclose(old_score, comparison["local_score"], rel_tol=1e-12):
        raise ValueError("Historical score differs from measured task costs.")
    expected["dqi.kmb"] = (NEW_DQI_SHA256, NEW_DQI_BYTES)
    return {"usage": "historical_reference_costs", "official_costs": {task: parent["official"]["tasks"][task]["cost"] for task in TASKS},
            "old_local_score": comparison["local_score"], "historical_artifact_sha256": parent["_sha256"],
            "expected_portfolio_components": expected,
            "reference_backend_equivalence": "Not established by historical runtime evidence; all new components are freshly verified."}


def check_current_backend(expected: dict, actual: dict) -> None:
    if not expected or actual != expected:
        raise ValueError("Current official backend/data hashes differ from the pinned local manifest.")


def combine_results(reference: dict, fresh: dict) -> dict:
    _check_full_result(fresh)
    if fresh["tasks"]["dfthcsos"]["verified_shots"] != 104704:
        raise ValueError("Chemistry full-strength shot count differs from the selected 1636-address portfolio.")
    actual = _records([record for task in TASKS for record in fresh["tasks"][task]["files"]])
    if actual != reference["expected_portfolio_components"] or fresh["tasks"]["dqi"]["raw"].get("circuit_sha256") != NEW_DQI_SHA256:
        raise ValueError("Fresh component manifests differ from the selected pinned portfolio.")
    rows = {task: {**fresh["tasks"][task], "fresh_this_run": True,
                   "official_reference_cost": reference["official_costs"][task]} for task in TASKS}
    new_score = math.exp(sum(math.log(reference["official_costs"][task] / rows[task]["cost"]) for task in TASKS) / 3)
    return {"status": "IMPROVEMENT_VERIFIED_FRESH_FULL" if new_score > reference["old_local_score"] else "NO_IMPROVEMENT",
            "verification_mode": MODE, "whole_submission_fresh_this_run": True,
            "complete_submission_verification": True, "cached_results_used": False, "tasks": rows,
            "old_local_score": reference["old_local_score"], "new_local_score": new_score,
            "local_score_multiplier": new_score / reference["old_local_score"], "observed_kaggle_score": None,
            "score_scope": "Fresh local salted candidate costs divided into the first full baseline's historical measured costs; not Kaggle scoring.",
            "reference_cost_source": reference, "salt": SALT}


def _base_builder():
    path = Path(__file__).with_name("prepare_kaggle_job.py")
    spec = importlib.util.spec_from_file_location("portfolio_base_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_job(root: Path, output_dir: Path, owner: str = "gengsr") -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", owner):
        raise ValueError("Invalid Kaggle owner.")
    root, output_dir = Path(root), Path(output_dir)
    base = _base_builder()
    blobs = parent_artifacts(root)
    hashes = {key: hashlib.sha256(blob).hexdigest() for key, blob in blobs.items()}
    historical_reference(decode_evidence(blobs, hashes))
    metric_dir = root / "external/notebooks/metric"
    metric_metadata = strict_json((metric_dir / "kernel-metadata.json").read_bytes())
    if not re.fullmatch(r"gcr\.io/kaggle-images/python@sha256:[0-9a-f]{64}", metric_metadata.get("docker_image", "")):
        raise ValueError("Current official image needs an immutable digest.")
    manifest = strict_json((root / "runs/input_manifest.json").read_bytes())
    backend = {}
    for record in manifest["files"]:
        prefix = "external/competition/"
        if record["path"].startswith(prefix) and Path(record["path"]).suffix in {".py", ".so", ".fcidump", ".npz"} and not record["path"].startswith(prefix + "sample_submission/"):
            backend[record["path"][len(prefix):]] = record["sha256"]
    if not backend:
        raise ValueError("Current official backend manifest is empty.")
    notebook = base.build_notebook((root / "scripts/verify_candidate.py").read_bytes(),
                                   (metric_dir / "google-quantum-ai-metric.ipynb").read_bytes(), metric_metadata, SALT)
    self_source = Path(__file__).read_bytes()
    payloads = {key: base64.b64encode(blob).decode("ascii") for key, blob in blobs.items()}
    setup = f'''
import importlib.util
PORTFOLIO_TOOL = TOOLS / "portfolio_tools.py"
source = base64.b64decode({base64.b64encode(self_source).decode('ascii')!r}, validate=True)
if hashlib.sha256(source).hexdigest() != {hashlib.sha256(self_source).hexdigest()!r}:
    raise ValueError("Portfolio helper source hash mismatch.")
PORTFOLIO_TOOL.write_bytes(source)
spec = importlib.util.spec_from_file_location("portfolio_tools", PORTFOLIO_TOOL)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
HISTORICAL_PAYLOADS = {payloads!r}
HISTORICAL_HASHES = {hashes!r}
HISTORY = P.decode_evidence({{key: base64.b64decode(blob, validate=True) for key, blob in HISTORICAL_PAYLOADS.items()}}, HISTORICAL_HASHES)
REFERENCE = P.historical_reference(HISTORY)
EXPECTED_BACKEND = {backend!r}
CURRENT_BACKEND = {{name: digest(DATA_ROOT / name) for name in EXPECTED_BACKEND}}
P.check_current_backend(EXPECTED_BACKEND, CURRENT_BACKEND)
ENVIRONMENT.update(verification_mode=P.MODE, all_three_tasks_fresh_this_run=True,
                   historical_reports_usage="historical_reference_costs_only",
                   current_backend_sha256=CURRENT_BACKEND, portfolio_source_sha256=digest(PORTFOLIO_TOOL))
save_json(RUNS / "environment.json", ENVIRONMENT)
for key, payload in HISTORICAL_PAYLOADS.items():
    (RUNS / ("historical_" + key + ".json")).write_bytes(base64.b64decode(payload, validate=True))
new_archives = [path for path in INPUT.rglob("submission.zip") if digest(path) == P.NEW_ARCHIVE_SHA256]
if len(new_archives) != 1:
    raise ValueError("Expected exactly one SHA-pinned Phuc V9 archive.")
NEW_SUBMISSION = WORK / "candidates/phuc_v9"
new_members = extract_submission(new_archives[0], NEW_SUBMISSION)
if digest(NEW_SUBMISSION / "dqi.kmb") != P.NEW_DQI_SHA256:
    raise ValueError("New DQI bytes differ from the pinned selected circuit.")
CANDIDATE = WORK / "candidates/portfolio"
shutil.copytree(PUBLIC_SUBMISSION, CANDIDATE)
shutil.copyfile(NEW_SUBMISSION / "dqi.kmb", CANDIDATE / "dqi.kmb")
for name, (expected_hash, expected_bytes) in REFERENCE["expected_portfolio_components"].items():
    if digest(CANDIDATE / name) != expected_hash or (CANDIDATE / name).stat().st_size != expected_bytes:
        raise ValueError("Selected portfolio component bytes differ: " + name)
save_json(RUNS / "portfolio_inputs.json", {{"new_kernel": P.NEW_KERNEL, "new_archive_sha256": digest(new_archives[0]),
           "new_members": new_members, "historical_artifact_sha256": HISTORICAL_HASHES,
           "current_backend_sha256": CURRENT_BACKEND, "verification_plan": "all_three_fresh_full"}})
print("Pinned portfolio assembled. Historical reports supply costs only; no historical PASS is reused.", flush=True)
'''
    execution = '''
output = RUNS / "fresh_portfolio"
command = [sys.executable, "-u", str(TOOLS / "verify_candidate.py"),
           "--data-root", str(DATA_ROOT), "--submission-dir", str(CANDIDATE),
           "--output-dir", str(output), "--salt", SALT,
           "--metric", str(TOOLS / "official_metric.ipynb"), "--tasks", *TASKS]
print("Starting all three fresh full verifications; no result reuse.", flush=True)
with (RUNS / "fresh_portfolio.console.log").open("w", encoding="utf-8") as log:
    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=3*60*60, check=False)
if process.returncode:
    raise RuntimeError("Fresh portfolio verification failed; no portfolio submission is packaged.")
fresh = P.strict_json((output / "results.json").read_bytes())
combined = P.combine_results(REFERENCE, fresh)
combined["fresh_result_sha256"] = digest(output / "results.json")
combined["current_backend_sha256"] = CURRENT_BACKEND
if combined["status"] == "IMPROVEMENT_VERIFIED_FRESH_FULL":
    package = WORK / "submission.zip"
    with zipfile.ZipFile(package, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(CANDIDATE.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(CANDIDATE).as_posix())
    expected = {record["path"]: record["sha256"] for row in combined["tasks"].values() for record in row["files"]}
    with zipfile.ZipFile(package) as archive:
        if set(archive.namelist()) != set(expected):
            raise ValueError("Portfolio archive member set differs from freshly verified components.")
        for name, expected_hash in expected.items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected_hash:
                    raise ValueError("Portfolio archive member bytes differ: " + name)
    combined["submission_zip_sha256"] = digest(package)
save_json(WORK / "portfolio_result.json", combined)
print(json.dumps({"status": combined["status"], "verification_mode": combined["verification_mode"],
                  "old_local_score": combined["old_local_score"], "new_local_score": combined["new_local_score"]}, indent=2), flush=True)
'''
    notebook["cells"] = notebook["cells"][:2] + [base.code_cell(setup), base.code_cell(execution)]
    base.validate_notebook(notebook)
    metadata = {"id": f"{owner}/{SLUG}", "title": "Quantum Portfolio Update", "code_file": SLUG + ".ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
                "enable_tpu": False, "enable_internet": False, "dataset_sources": [], "model_sources": [],
                "kernel_sources": [base.PUBLIC_KERNEL, NEW_KERNEL], "competition_sources": [base.COMPETITION],
                "docker_image": metric_metadata["docker_image"], "docker_image_pinning_type": "original"}
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, document in ((metadata["code_file"], notebook), ("kernel-metadata.json", metadata)):
        (output_dir / filename).write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return {"status": "PREPARED", "kernel_id": metadata["id"], "uploaded": False,
            "directory": str(output_dir.resolve()), "verification_mode": MODE, "historical_artifact_sha256": hashes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--owner", default="gengsr")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    print(json.dumps(prepare_job(root, args.output_dir or root / "kaggle" / SLUG, args.owner), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
