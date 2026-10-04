"""Run the selected benchmark in this checkout's isolated, dedicated .venv.

Setup/network work finishes before bench.runner starts its experiment clock.
The original reference measurements used CPython 3.14.0; Python >=3.11 is the
portability target, not a claim of numerically identical timings across versions.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import site
import stat
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
MIN_PYTHON = (3, 11)
DILL_VERSION = "0.4.1"
SETUP_TIMEOUT_SECONDS = 300

_PROBE = r"""
import importlib.metadata, json, site, sys
result = {
    "prefix": sys.prefix, "base_prefix": sys.base_prefix,
    "python": list(sys.version_info[:3]), "isolated": sys.flags.isolated,
    "dont_write_bytecode": sys.dont_write_bytecode,
    "user_site": site.ENABLE_USER_SITE,
    "dill_version": None, "dill_import_version": None, "dill_path": None,
}
try:
    result["dill_version"] = importlib.metadata.version("dill")
    import dill
    result["dill_import_version"] = dill.__version__
    result["dill_path"] = dill.__file__
except Exception as error:
    result["dependency_error"] = type(error).__name__ + ": " + str(error)
print(json.dumps(result))
"""


class BootstrapError(RuntimeError):
    """An actionable setup failure; no benchmark has been started."""


def _seconds(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("budget-seconds must be a number") from error
    if not math.isfinite(number) or not 0 < number <= 7200:
        raise argparse.ArgumentTypeError("budget-seconds must be finite and in (0, 7200]")
    return number


def _mib(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("max-mib must be an integer") from error
    if not 1 <= number <= 128:
        raise argparse.ArgumentTypeError("max-mib must be in [1, 128]")
    return number


def _seed_offset(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed-offset must be an integer") from error
    if not 0 <= number <= 1_000_000_000:
        raise argparse.ArgumentTypeError("seed-offset must be in [0, 1000000000]")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--preset", choices=("idle-primary",), default="idle-primary")
    result.add_argument("--condition", choices=("idle", "busy", "unspecified"), default="unspecified",
                        help="user-declared host condition; not an interference-free certificate")
    result.add_argument("--output", type=Path, help="new output directory; existing directories are not reused")
    result.add_argument("--budget-seconds", type=_seconds, default=600.0)
    result.add_argument("--max-mib", type=_mib, default=16)
    result.add_argument("--seed-offset", type=_seed_offset, default=0)
    result.add_argument("--setup-only", action="store_true",
                        help="prepare the dedicated environment without running an experiment")
    return result


def venv_python(root: Path, *, platform: str | None = None) -> Path:
    platform = os.name if platform is None else platform
    return root / ".venv" / ("Scripts/python.exe" if platform == "nt" else "bin/python")


def _setup_command(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, check=True, timeout=SETUP_TIMEOUT_SECONDS,
                              capture_output=capture, text=True)
    except subprocess.TimeoutExpired as error:
        raise BootstrapError("environment setup exceeded 300 seconds; experiment not started") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()[-2000:]
        raise BootstrapError(f"environment command failed (exit {error.returncode}). {detail}") from error
    except OSError as error:
        raise BootstrapError(f"cannot execute environment command: {error}") from error


def _reject_environment_link(root: Path) -> None:
    directory = root / ".venv"
    if directory.is_symlink() or (hasattr(directory, "is_junction") and directory.is_junction()):
        raise BootstrapError(".venv must be a dedicated directory, not a symlink or junction")
    # Path.is_junction was added after Python 3.11. Windows reparse attributes
    # also detect a dangling junction before any creation could follow it.
    if os.name == "nt":
        try:
            attributes = directory.lstat().st_file_attributes
        except FileNotFoundError:
            return
        if attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise BootstrapError(".venv must not be a Windows reparse point")


def _check_layout(root: Path) -> Path:
    _reject_environment_link(root)
    directory = root / ".venv"
    python = venv_python(root)
    config = directory / "pyvenv.cfg"
    if not directory.is_dir() or not python.is_file() or not config.is_file():
        raise BootstrapError("existing .venv is incomplete; inspect/recreate it manually, then retry")
    settings = {}
    for line in config.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            settings[key.strip().lower()] = value.strip().lower()
    if settings.get("include-system-site-packages") != "false":
        raise BootstrapError(".venv must disable system-site-packages; it was not modified")
    return python


def _probe(python: Path, root: Path) -> dict:
    completed = _setup_command([str(python), "-I", "-B", "-c", _PROBE], capture=True)
    try:
        result = json.loads(completed.stdout)
        directory = (root / ".venv").resolve()
        valid = (Path(result["prefix"]).resolve() == directory
                 and Path(result["base_prefix"]).resolve() != directory
                 and tuple(result["python"][:2]) >= MIN_PYTHON
                 and result["isolated"] == 1 and result["dont_write_bytecode"] is True
                 and result["user_site"] is False)
    except (ValueError, TypeError, KeyError) as error:
        raise BootstrapError("dedicated interpreter returned an invalid setup receipt") from error
    if not valid:
        raise BootstrapError(".venv interpreter is not Python>=3.11 in the required isolated environment")
    return result


def _dependency_ready(receipt: dict, root: Path) -> bool:
    package = receipt.get("dill_path")
    return (receipt.get("dill_version") == DILL_VERSION
            and receipt.get("dill_import_version") == DILL_VERSION
            and isinstance(package, str)
            and Path(package).resolve().is_relative_to((root / ".venv").resolve()))


def ensure_environment(root: Path) -> Path:
    """Prepare only .venv. A matching dependency never triggers another install."""
    directory = root / ".venv"
    _reject_environment_link(root)
    if not directory.exists():
        print("Creating dedicated .venv (outside experiment timing).", flush=True)
        try:
            _setup_command([sys.executable, "-I", "-B", "-m", "venv", str(directory)])
        except BootstrapError as error:
            hint = ""
            if os.name != "nt":
                version = f"{sys.version_info[0]}.{sys.version_info[1]}"
                hint = (" On Linux, missing ensurepip/venv support must be installed by you "
                        "using your distribution's package manager. On Ubuntu/Debian this is "
                        f"typically python3-venv or the matching python{version}-venv package "
                        "(for example python3.12-venv for Python 3.12). No system packages "
                        "were installed by this script.")
            raise BootstrapError(f"Virtual environment creation failed: {error}.{hint} "
                                 "If a partial .venv was created, inspect/remove only that "
                                 "directory before retrying; the script will not overwrite it.") from error
    python = _check_layout(root)
    receipt = _probe(python, root)
    if not _dependency_ready(receipt, root):
        lock = root / "requirements.lock"
        if not lock.is_file():
            raise BootstrapError("requirements.lock is missing; no dependency was installed")
        print("Installing hash-locked dill in .venv (outside experiment timing).", flush=True)
        _setup_command([str(python), "-I", "-B", "-m", "pip", "--isolated",
                        "--disable-pip-version-check", "--no-input", "install", "--no-cache-dir",
                        "--require-hashes", "--only-binary=:all:", "--no-deps", "--force-reinstall",
                        "--index-url", "https://pypi.org/simple", "-r", str(lock)])
        receipt = _probe(python, root)
        if not _dependency_ready(receipt, root):
            raise BootstrapError("locked dill0.4.1 did not load from .venv after installation")
    return python


def _already_isolated(root: Path) -> bool:
    return (Path(sys.prefix).resolve() == (root / ".venv").resolve()
            and sys.flags.isolated == 1 and sys.dont_write_bytecode
            and site.ENABLE_USER_SITE is False)


def _run_benchmark(root: Path, arguments: list[str]) -> int:
    # -I removes user/current-directory imports; add only this repository's roots.
    sys.path[:0] = [str(root / "src"), str(root / "vendor"), str(root)]
    runner = importlib.import_module("bench.runner")
    return int(runner.main(arguments))


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    options = parser().parse_args(arguments)  # --help exits before setup/imports.
    try:
        if sys.version_info[:2] < MIN_PYTHON:
            raise BootstrapError("Python >=3.11 is required; reference measurements used CPython 3.14.0")
        python = ensure_environment(ROOT)
        if options.setup_only:
            print(f"Environment ready: {python}; no experiment started.")
            return 0
        if not _already_isolated(ROOT):
            # Preserve arguments as a list, including spaces; no shell/activation.
            return subprocess.run([str(python), "-I", "-B", str(ROOT / "run_experiment.py"),
                                   *arguments], check=False).returncode
        return _run_benchmark(ROOT, arguments)
    except KeyboardInterrupt:
        print("Interrupted; no experiment is automatically retried.", file=sys.stderr)
        return 130
    except (BootstrapError, OSError, ImportError) as error:
        print(f"Execution error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
