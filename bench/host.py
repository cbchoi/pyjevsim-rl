"""Optional, read-only host endpoints; no sampler, load generator, or process control."""
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import sys
import time


def _system_cpu():
    if sys.platform.startswith("linux"):
        fields = Path("/proc/stat").read_text().splitlines()[0].split()
        if fields[0] != "cpu" or len(fields) < 9:
            raise ValueError("aggregate /proc/stat CPU row unavailable")
        values = [int(value) for value in fields[1:9]]  # guest values overlap user/nice.
        ticks = os.sysconf("SC_CLK_TCK")
        return {"source": "linux-proc-stat", "total_seconds": sum(values) / ticks,
                "idle_seconds": (values[3] + values[4]) / ticks,
                "iowait_seconds": values[4] / ticks, "idle_includes_iowait": True}
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class FileTime(ctypes.Structure):
            _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        function = kernel.GetSystemTimes
        function.argtypes = [ctypes.POINTER(FileTime)] * 3
        function.restype = wintypes.BOOL
        idle, system, user = FileTime(), FileTime(), FileTime()
        if not function(ctypes.byref(idle), ctypes.byref(system), ctypes.byref(user)):
            raise ctypes.WinError(ctypes.get_last_error())
        seconds = lambda value: ((value.high << 32) | value.low) / 10_000_000
        return {"source": "windows-GetSystemTimes", "total_seconds": seconds(system) + seconds(user),
                "idle_seconds": seconds(idle), "iowait_seconds": None,
                "idle_includes_iowait": None, "processor_group_coverage": "OS API scope; all groups not independently verified"}
    return None


def _disk_io():
    if not sys.platform.startswith("linux"):
        return None
    devices = {}
    for line in Path("/proc/diskstats").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 14:
            devices[parts[2]] = {"read_bytes": int(parts[5]) * 512, "write_bytes": int(parts[9]) * 512}
    return devices  # Do not sum devices: partitions and stacked devices may overlap.


def snapshot():
    result = {"monotonic_seconds": time.monotonic(), "observer_pid": os.getpid(),
              "observer_process_cpu_seconds": time.process_time(), "errors": {}}
    for key, reader in (("system_cpu", _system_cpu), ("disk_io", _disk_io)):
        try:
            result[key] = reader()
        except (OSError, ValueError, AttributeError) as exc:
            result[key] = None
            result["errors"][key] = f"{type(exc).__name__}: {exc}"
    result["coverage"] = {"system_cpu": result["system_cpu"] is not None,
                          "disk_io": result["disk_io"] is not None, "external_process_attribution": False}
    return result


def difference(before, after):
    elapsed = after["monotonic_seconds"] - before["monotonic_seconds"]
    cpu_before, cpu_after = before.get("system_cpu"), after.get("system_cpu")
    fraction = None
    if elapsed > 0 and cpu_before and cpu_after and cpu_before["source"] == cpu_after["source"]:
        total = cpu_after["total_seconds"] - cpu_before["total_seconds"]
        idle = cpu_after["idle_seconds"] - cpu_before["idle_seconds"]
        if total > 0 and 0 <= idle <= total:
            fraction = 1 - idle / total
    disks = None
    if before.get("disk_io") is not None and after.get("disk_io") is not None:
        disks = {}
        for name in sorted(set(before["disk_io"]) & set(after["disk_io"])):
            changes = {key: after["disk_io"][name][key] - before["disk_io"][name][key]
                       for key in ("read_bytes", "write_bytes")}
            disks[name] = changes if all(value >= 0 for value in changes.values()) else None
    observer = None
    if before.get("observer_pid") == after.get("observer_pid"):
        value = after["observer_process_cpu_seconds"] - before["observer_process_cpu_seconds"]
        observer = value if value >= 0 else None
    return {"elapsed_wall_seconds": elapsed, "aggregate_cpu_busy_fraction": fraction,
            "aggregate_cpu_observed": fraction is not None, "disk_io_by_device": disks,
            "observer_process_cpu_seconds": observer, "observer_pid": before.get("observer_pid"),
            "external_interference_identified": False,
            "limitations": "system totals include benchmark and other processes; observer CPU is parent, not worker; disk devices overlap; wall-minus-CPU is not a causal interference measure"}


def _memory():
    if sys.platform.startswith("linux"):
        rows = dict((line.split(":", 1)[0], int(line.split()[1]) * 1024)
                    for line in Path("/proc/meminfo").read_text().splitlines() if len(line.split()) >= 2)
        return {"total_bytes": rows.get("MemTotal"), "available_bytes": rows.get("MemAvailable"), "source": "proc-meminfo"}
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Memory(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong) for name in ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
        value = Memory()
        value.length = ctypes.sizeof(value)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            raise OSError("GlobalMemoryStatusEx failed")
        return {"total_bytes": value.total, "available_bytes": value.available, "source": "GlobalMemoryStatusEx"}
    return None


def metadata(storage_path=None):
    errors = {}
    try:
        memory = _memory()
    except (OSError, ValueError, AttributeError) as exc:
        memory, errors["memory"] = None, str(exc)
    storage = None
    if storage_path is not None:
        try:
            path = Path(storage_path).resolve()
            values = shutil.disk_usage(path)
            storage = {"path": str(path), "total_bytes": values.total, "used_bytes": values.used, "free_bytes": values.free,
                       "scope": "capacity only; not storage throughput or device attribution"}
        except OSError as exc:
            errors["storage"] = str(exc)
    dependencies = {}
    for name in ("dill", "pyjevsim", "pyjevsim-rl"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return {"os": platform.platform(), "machine": platform.machine(), "processor": platform.processor() or None,
            "logical_cpus": os.cpu_count(), "python": sys.version, "python_executable": sys.executable,
            "dependencies": dependencies, "memory": memory, "storage": storage, "errors": errors,
            "condition_label_is_clean_host_proof": False, "whole_lifetime_memory_complete": False}
