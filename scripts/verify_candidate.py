"""Fresh, full-strength verification using the original competition readers and verifiers."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import stat
import sys
import time
import traceback
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any


TASKS = ("invsqrt", "dqi", "dfthcsos")
DEFAULT_SALT = "quantum-local-full-verification-20261025-v1"
DEFAULT_METRIC = Path(__file__).resolve().parents[1] / "external/notebooks/metric/google-quantum-ai-metric.ipynb"


def _one(root: Path, pattern: str, directory: bool = False) -> Path:
    matches = [
        path for path in root.rglob(pattern)
        if path.is_dir() == directory and "__MACOSX" not in path.relative_to(root).parts
    ]
    if len(matches) != 1:
        raise ValueError(f"The submission must contain exactly one {pattern}; found {len(matches)}.")
    return matches[0]


def collect_submission_files(submission_dir: Path, tasks: tuple[str, ...] = TASKS) -> dict[str, list[Path]]:
    """Check the selected tasks' archive structure and return their inputs in verifier order."""
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(TASKS):
        raise ValueError(f"tasks must be a nonempty, unique selection from {TASKS}")
    root = Path(submission_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Submission directory does not exist: {root}")
    files: dict[str, list[Path]] = {}
    for task in tasks:
        if task != "dfthcsos":
            files[task] = [_one(root, f"{task}.km[xb]")]
        else:
            folder = _one(root, "dfthcsos", directory=True)
            angles = list(folder.glob("angle_k_*.km[xb]"))
            by_stem = {path.stem: path for path in angles}
            stems = [f"angle_k_{i}" for i in range(len(angles))]
            if not stems or sorted(by_stem) != sorted(stems):
                raise ValueError("dfthcsos must contain angle_k_0 ... angle_k_<n-1>, each once, without gaps.")
            files[task] = [by_stem[stem] for stem in stems] + [_one(folder, "submission.npz")]
        for path in files[task]:
            status = path.lstat()
            if not stat.S_ISREG(status.st_mode) or status.st_nlink > 1 or not path.resolve().is_relative_to(root):
                raise ValueError(f"{path} must be a regular file, not a link or a special file.")
    return files


def load_metric(path: Path) -> types.ModuleType:
    """Load code cells of the unchanged official metric, or its extracted Python source."""
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    if path.suffix == ".ipynb":
        notebook = json.loads(source)
        source = "\n\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    module = types.ModuleType("quantum_official_metric")
    module.__file__ = str(path.resolve())
    exec(compile(source, str(path), "exec"), module.__dict__)
    for name in ("read_circuit", "read_npz"):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"Official metric is missing {name}.")
    return module


def load_backend(data_root: Path) -> SimpleNamespace:
    """Use the pinned competition backend; never silently prefer a different local install."""
    data_root = Path(data_root).resolve()
    sys.path[:0] = [str(data_root / "kickmix"), str(data_root / "verifiers")]
    names = {
        "kickmix": "kickmix", "invsqrt": "invsqrt.verify_submission_invsqrt",
        "dqi": "dqi.verify_submission_dqi", "dfthcsos": "dfthcsos.verify",
        "reference": "dfthcsos.scoreinfra.reference",
    }
    modules = {key: importlib.import_module(name) for key, name in names.items()}
    for name, module in modules.items():
        if not Path(module.__file__).resolve().is_relative_to(data_root):
            raise RuntimeError(f"{name} was imported from {module.__file__}, outside official data {data_root}; start a fresh process.")
    return SimpleNamespace(**modules)


def _prepare_verifications(files: dict[str, list[Path]], metric: Any, backend: Any, salt: str) -> dict[str, Any]:
    prepared: dict[str, Any] = {}
    for task, paths in files.items():
        if task == "invsqrt":
            circuit = metric.read_circuit(paths[0], 1)
            spec = backend.invsqrt.ProblemSpecInvSqrt.for_n_p(10)
            prepared[task] = lambda c=circuit, s=spec: backend.invsqrt.verify(c, s, salt=salt)
        elif task == "dqi":
            spec = backend.dqi.ProblemSpecDQI.benchmark_instance()
            circuit = metric.read_circuit(paths[0], spec.n + spec.m)
            prepared[task] = lambda c=circuit, s=spec: backend.dqi.verify(c, s, salt=salt)
        else:
            angles = [metric.read_circuit(path, backend.dfthcsos.QUBIT_CAP) for path in paths[:-1]]
            orbitals = backend.reference.ORBITALS
            npz_spec = {
                "U": ("f", ("R", "B", orbitals)), "W": ("f", ("R", "B", "C")),
                "H2_I": ("f", ("R", "C")), "H1_I": ("f", ()),
                "Hsym_lower": ("f", (orbitals * (orbitals + 1) // 2,)),
                "tree_edges": ("iu", (orbitals - 1, 2)),
            }
            if set(npz_spec) != {*backend.reference.FACTORIZATION_KEYS, "tree_edges"}:
                raise ValueError("Official chemistry factorization keys changed; review preflight schema.")
            factorization = metric.read_npz(paths[-1], npz_spec)
            edges = factorization.pop("tree_edges")
            prepared[task] = lambda a=angles, f=factorization, e=edges: backend.dfthcsos.verify(
                a, f, e, salt=salt, shots_per_address=backend.dfthcsos.SHOTS_PER_ADDRESS
            )
    return prepared


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": path.relative_to(root).as_posix(), "sha256": digest, "bytes": path.stat().st_size}


def _json_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=_json_value)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


def run_verification(
    data_root: Path, submission_dir: Path, output_dir: Path,
    salt: str = DEFAULT_SALT, metric_path: Path = DEFAULT_METRIC,
    tasks: tuple[str, ...] = TASKS,
) -> int:
    """Run every selected verifier fresh at full strength; write results and return 0 or 1."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "status": "RUNNING", "verification_mode": "fresh_full", "cached_results_used": False,
        "complete_submission_verification": False, "tasks": {},
        "data_root": str(Path(data_root).resolve()), "submission_dir": str(Path(submission_dir).resolve()),
        "salt": salt, "score_scope": "local salted verification; not a Kaggle Public or Private score",
        "python": sys.version, "platform": platform.platform(),
    }
    _write_json(output_dir / "results.json", summary)
    try:
        if not isinstance(salt, str) or len(salt) < 32:
            raise ValueError("Local verification salt must contain at least 32 characters.")
        files = collect_submission_files(Path(submission_dir), tasks)
        root = Path(submission_dir).resolve()
        records = {task: [_file_record(path, root) for path in paths] for task, paths in files.items()}
        with (output_dir / "setup.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            metric = load_metric(Path(metric_path))
            backend = load_backend(Path(data_root))
            prepared = _prepare_verifications(files, metric, backend, salt)
        summary["kickmix_path"] = str(backend.kickmix.__file__)
        summary["metric_sha256"] = _file_record(Path(metric_path).resolve(), Path(metric_path).resolve().parent)["sha256"]
        summary["numpy"] = importlib.metadata.version("numpy")
        for task, verify in prepared.items():
            print(f"[{task}] full verification started", flush=True)
            started = time.monotonic()
            report: dict[str, Any] = {"status": "RUNNING", "files": records[task], "fresh_verification": True}
            with (output_dir / f"{task}.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    raw = verify()
                    cost = float(raw["score"])
                    if raw.get("status") != "PASS" or not math.isfinite(cost) or cost <= 0:
                        raise ValueError("Verifier must return PASS with a finite, positive cost.")
                    report.update(
                        status="PASS", cost=cost, raw=raw,
                        num_qubits=raw["num_qubits"], verified_shots=raw["verified_shots"],
                    )
                except Exception as error:
                    traceback.print_exc()
                    report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
            report["seconds"] = time.monotonic() - started
            _write_json(output_dir / f"{task}.json", report)
            summary["tasks"][task] = report
            _write_json(output_dir / "results.json", summary)
            print(f"[{task}] {report['status']} ({report['seconds']:.1f}s)", flush=True)
            if report["status"] != "PASS":
                raise RuntimeError(f"{task} failed verification; see {task}.log")
        complete = set(tasks) == set(TASKS)
        summary.update(status="PASS" if complete else "PARTIAL_PASS", complete_submission_verification=complete)
    except Exception as error:
        summary.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        with (output_dir / "error.log").open("w", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        print(summary["error"], file=sys.stderr, flush=True)
        _write_json(output_dir / "results.json", summary)
        return 1
    _write_json(output_dir / "results.json", summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--salt", default=DEFAULT_SALT)
    parser.add_argument("--metric", type=Path, default=DEFAULT_METRIC)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    args = parser.parse_args(argv)
    return run_verification(args.data_root, args.submission_dir, args.output_dir, args.salt, args.metric, tuple(args.tasks))


if __name__ == "__main__":
    raise SystemExit(main())
