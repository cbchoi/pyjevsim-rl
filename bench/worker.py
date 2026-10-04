"""One owned, isolated N/C1 arm; no installation, retry or network workflow."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
from pathlib import Path
import platform
import sys
import time


class ShortLog(io.TextIOBase):
    def __init__(self):
        self.text = ""
        self.discarded_characters = 0

    def write(self, value):
        value = str(value)
        kept = value[:max(0, 8192 - len(self.text))]
        self.text += kept
        self.discarded_characters += len(value) - len(kept)
        return len(value)


def provenance(root, modules):
    """Actual loaded repository sources, not a complete dependency closure."""
    files = {}
    for name, module in tuple(sys.modules.items()):
        if (name.split(".")[0] not in ("pyjevsim", "pyjevsim_bridge", "rti1516e", "continuation_study")
                and not name.startswith("bench.research")):
            continue
        location = getattr(module, "__file__", None)
        if location is None:
            continue
        path = Path(location).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"loaded research source is outside repository: {name}")
        files[name] = {"path": path.relative_to(root).as_posix(),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for name in ("worker.py", "runner.py", "design.py", "host.py", "analyze.py"):
        path = root / "bench" / name
        files["bench." + name] = {"path": path.relative_to(root).as_posix(),
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    provider = importlib.import_module("pyjevsim_bridge.rl.continuation.engine_pyjevsim")
    # Native qualification applies to both methods, outside application timing.
    identity = provider.source_identity()
    dill = importlib.import_module("dill")
    return {"python": sys.version, "implementation": platform.python_implementation(),
        "platform": platform.platform(), "dill_version": getattr(dill, "__version__", None),
        "dill_init_sha256": hashlib.sha256(Path(dill.__file__).read_bytes()).hexdigest(),
        "source_files": files, "native_source_identity": identity,
        "source_inventory_complete": False, "paths": "repository-relative",
        "runtime_source_roots": ["src", "vendor", "bench"],
        "startup_import_provenance_in_application_endpoint": False}


def main(argv=None):
    worker_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_bytes())
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "src"), str(root / "vendor"), str(root / "bench"), str(root)]
    spec, case = request["spec"], request["case"]
    log = ShortLog()
    try:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            run = importlib.import_module("continuation_study.run")
            modules = run.load_runtime_modules()
            kind = request.get("experiment_kind", "idle-primary")
            if kind == "runtime-cost-v1":
                extension = importlib.import_module("bench.research.cost")
            elif kind == "idle-primary":
                extension = None
            else:
                raise ValueError("unknown allowlisted experiment kind")
            identity = provenance(root, modules)
            budget = run.Budget(request["campaign"], request["remaining_seconds"], request["max_bytes"])
            budget.started = worker_started
            budget.written = request["retained_bytes"]
            entered = {}

            def observed_backend(*args, **kwargs):
                entered["wall"] = time.perf_counter()
                entered["cpu"] = time.process_time()
                return run.Backend(*args, **kwargs)

            internal = dict(spec, method="C" if spec["method"] == "C1" else spec["method"])
            if extension is None:
                row, projection = run.execute_arm(internal, case, request["campaign"], budget, modules,
                                                  backend_factory=observed_backend)
            else:
                row, projection = extension.execute_arm(run, internal, case, request["campaign"],
                    budget, modules, backend_factory=observed_backend)
            cpu_end, wall_end = time.process_time(), time.perf_counter()
            row.update(spec)
            row.update(application_cpu_seconds=None if not entered else cpu_end - entered["cpu"],
                cpu_scope_wall_seconds=None if not entered else wall_end - entered["wall"],
                cpu_endpoint="backend-factory entry through execute_arm return; process CPU only",
                cpu_endpoint_matches_application_wall=False,
                cpu_instrumentation="two start/end clock pairs; identical across methods; overhead not subtracted",
                profile_overhead_included=spec["purpose"] == "counting", event_counting_enabled=spec["purpose"] == "counting",
                worker_elapsed_before_transport_seconds=time.perf_counter() - worker_started,
                worker_peak_storage_observed_bytes=budget.peak_observed_bytes)
            packet = {"row": row, "projection": projection, "provenance": identity}
    except BaseException as exc:
        packet = {"row": {**spec, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            "error_phase": "worker-import-or-outside-arm", "config_sha256": case["config_sha256"],
            "seed": case["seed"], "application_wall_seconds": None, "application_cpu_seconds": None,
            "cpu_scope_wall_seconds": None, "cleanup_confirmed": None, "cleanup_errors": [],
            "memory_complete": False, "whole_lifetime_peak_bytes": None},
            "projection": [], "provenance": None}
    packet["worker_log"] = {"text": log.text, "discarded_characters": log.discarded_characters}
    payload = json.dumps(packet, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > request["max_transport_bytes"]:
        packet["row"].update(status="failed", error="result transport exceeded declared cap",
                             transport_bytes_required=len(payload))
        packet["projection"] = []
        payload = json.dumps(packet, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()
    return 0 if packet["row"]["status"] == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())
