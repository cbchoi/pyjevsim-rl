"""Record a byte-preserved, explicitly scoped import; not an execution gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", required=True, type=Path)
    parser.add_argument("--native-root", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    repo = args.source_repo.resolve()
    native = args.native_root.resolve()
    groups = ((root / "src" / "pyjevsim_bridge", repo / "pysdk" / "pyjevsim_bridge"),
              (root / "src" / "rti1516e", repo / "pysdk" / "rti1516e"),
              (root / "vendor" / "pyjevsim", native),
              (root / "bench" / "continuation_study", repo / "engineering" / "specifications" / "pyjevsim-rl" / "continuation_study"))
    records = []
    for target, source in groups:
        for path in sorted(target.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            data = path.read_bytes()
            original = source / path.relative_to(target)
            if data != original.read_bytes():
                raise ValueError(f"import bytes differ: {path.relative_to(root)}")
            records.append({"path": path.relative_to(root).as_posix(), "bytes": len(data),
                            "sha256": hashlib.sha256(data).hexdigest()})
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    result = {"schema": "portable-benchmark-scoped-source-import-v1", "original_repository": "gorti",
        "original_head": commit, "import_kind": "selected-working-tree-files-not-clean-commit-export",
        "original_git_index_modified": False, "native_metadata_label": "2.1.1",
        "native_identity_basis": "actual preserved source bytes; engine provider's twelve pins remain unchanged",
        "excluded": ["git history", "old experiment data", "credentials", "caches", "generated network stubs"],
        "records": records, "file_count": len(records), "bytes": sum(row["bytes"] for row in records)}
    target = root / "provenance" / "source-import.json"
    target.parent.mkdir(exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Recorded {len(records)} original files without changing their bytes.")


if __name__ == "__main__":
    main()
