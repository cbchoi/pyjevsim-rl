"""Build wrapper: regenerate _generated/ from proto/rti/v1/*.proto.

This module provides the Python implementation used by ``py-codegen`` in
the repo-root ``Makefile`` and by ``pysdk/Makefile.codegen``. Both entry
points must remain logically equivalent.

Usage:

    python -m rti1516e._proto

Outputs land in ``pysdk/rti1516e/_generated/`` (gitignored). The directory
and an empty ``__init__.py`` are created on demand so the package is
importable even before grpcio-tools is installed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTO_DIR = REPO_ROOT / "proto"
OUT_DIR = REPO_ROOT / "pysdk" / "rti1516e" / "_generated"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "__init__.py").touch()
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{PROTO_DIR}",
        f"--python_out={OUT_DIR}",
        f"--grpc_python_out={OUT_DIR}",
        f"--pyi_out={OUT_DIR}",
        *[str(p) for p in sorted((PROTO_DIR / "rti" / "v1").glob("*.proto"))],
    ]
    print(" ".join(cmd))
    # cmd is built from constants + interpreter path + repo-local proto files;
    # no untrusted input is involved.
    subprocess.run(cmd, check=True)  # noqa: S603


if __name__ == "__main__":
    main()
