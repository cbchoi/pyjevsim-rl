"""Versioned checkpoint byte storage and source-independent portable closure.

This file deliberately uses only the standard library. An experiment may load
the exact same frozen helper for an older installed runner without importing or
changing that runner. These checks establish byte closure, not learner semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

STANDALONE_PROFILE = "standalone-v1"
SHARED_PROFILE = "shared-run-v2"
SHARED_SCHEMA = "local-runner-checkpoint-v2"
_V1_SCHEMA = "local-runner-checkpoint-v1"
_FIELDS = {
    "schema_version",
    "config_sha256",
    "source_identity",
    "runtime_identity",
    "objective_id",
    "run_id",
    "generation",
    "completed_updates",
    "completed_transitions",
    "next_job_index",
    "seed_schedule_sha256",
    "active_policy_reference",
    "learner_recovery_state",
    "artifacts",
    "episode_dispositions_sha256",
    "checkpoint_sha256",
}
_FILE_FIELDS = {"path", "size_bytes", "sha256"}


class StorageContractError(ValueError):
    """A storage profile or its immutable transitive byte closure is invalid."""


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise StorageContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> object:
    raise StorageContractError(f"nonfinite JSON token: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise StorageContractError("nonfinite JSON numeric overflow")
    return number


def decode(body: bytes) -> Any:
    try:
        return json.loads(
            body, object_pairs_hook=_pairs, parse_constant=_nonfinite, parse_float=_finite_float
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise StorageContractError("malformed storage JSON") from error


def profile(value: object) -> str:
    if value not in (STANDALONE_PROFILE, SHARED_PROFILE):
        raise StorageContractError("unknown checkpoint storage profile")
    return str(value)


def _sha(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(letter not in "0123456789abcdef" for letter in value)
    ):
        raise StorageContractError("invalid storage SHA256")
    return value


def _plain(path: Path) -> None:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise StorageContractError("storage symlink/reparse point is forbidden")
    if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
        raise StorageContractError("storage hard-linked file is forbidden")


def contained_file(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise StorageContractError("invalid archive-relative path")
    parts = relative.split("/")
    if any(part in ("", ".", "..") or part.endswith((".", " ")) for part in parts) or (
        PurePosixPath(relative).is_absolute()
    ):
        raise StorageContractError("storage path escapes archive")
    _plain(root)
    current = root
    for part in parts:
        current = current / part
        _plain(current)
    if not current.is_file():
        raise StorageContractError("storage dependency is not a regular file")
    return current


def atomic_write(path: Path, body: bytes) -> None:
    """Durable exclusive publication; no overwrite and no persistent hardlink."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".pending-" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    if temporary.read_bytes() != body:
        raise StorageContractError("storage write readback differs")
    os.link(temporary, path)
    temporary.unlink()


def file_entry(relative: str, body: bytes) -> dict[str, Any]:
    return {"path": relative, "size_bytes": len(body), "sha256": digest(body)}


def checked_bytes(root: Path, entry: object, *, artifact: bool = False) -> bytes:
    fields = _FILE_FIELDS | ({"reference"} if artifact else set())
    if not isinstance(entry, dict) or set(entry) != fields:
        raise StorageContractError("storage file entry fields differ")
    size = entry["size_bytes"]
    if type(size) is not int or size < 1:
        raise StorageContractError("storage length must be a positive integer")
    expected = _sha(entry["sha256"])
    body = contained_file(root, entry["path"]).read_bytes()
    if len(body) != size or digest(body) != expected:
        raise StorageContractError("storage dependency bytes differ")
    return body


def checkpoint_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).absolute()
    if path.is_dir():
        path = path / "manifest.json"
    _plain(path)
    return path


def archive_root(path: Path) -> Path:
    if path.name != "manifest.json" or path.parent.parent.name != "checkpoints":
        raise StorageContractError("archive checkpoint layout differs")
    return path.parent.parent.parent


def artifact_root(path: Path, manifest: dict[str, Any]) -> Path:
    return archive_root(path) if manifest["schema_version"] == SHARED_SCHEMA else path.parent


def read_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    """Validate manifest, policy byte references, episodes and v2 raw closure."""
    path = checkpoint_path(checkpoint)
    raw = decode(path.read_bytes())
    if not isinstance(raw, dict):
        raise StorageContractError("checkpoint must be an object")
    shared = raw.get("schema_version") == SHARED_SCHEMA
    if raw.get("schema_version") not in (_V1_SCHEMA, SHARED_SCHEMA):
        raise StorageContractError("unknown checkpoint schema")
    fields = _FIELDS | ({"storage_profile", "raw_batches"} if shared else set())
    if set(raw) != fields:
        raise StorageContractError("checkpoint fields differ from storage schema")
    unsigned = {key: value for key, value in raw.items() if key != "checkpoint_sha256"}
    if _sha(raw["checkpoint_sha256"]) != digest(canonical(unsigned)):
        raise StorageContractError("checkpoint digest differs")
    count = raw["completed_updates"]
    if type(count) is not int or count < 0:
        raise StorageContractError("checkpoint update count differs")
    if shared and (
        raw["storage_profile"] != SHARED_PROFILE or path.parent.name != f"update-{count:06d}"
    ):
        raise StorageContractError("shared profile/layout differs")
    entries = raw["artifacts"]
    if not isinstance(entries, list) or len(entries) != count + 1:
        raise StorageContractError("checkpoint omits historical policies")
    root = artifact_root(path, raw)
    seen: set[str] = set()
    for version, entry in enumerate(entries):
        body = checked_bytes(root, entry, artifact=True)
        if entry["path"] in seen:
            raise StorageContractError("duplicate artifact path")
        seen.add(entry["path"])
        if shared and entry["path"] != f"blobs/{entry['sha256']}.json":
            raise StorageContractError("shared blob path is not content addressed")
        policy = decode(body)
        reference = entry["reference"]
        if (
            not isinstance(policy, dict)
            or not isinstance(reference, dict)
            or policy.get("policy_version") != version
            or reference.get("policy_version") != version
            or policy.get("sha256") != reference.get("sha256")
        ):
            raise StorageContractError("policy identity/version differs")
    if raw["active_policy_reference"] != entries[-1]["reference"]:
        raise StorageContractError("active policy is not final history entry")
    episodes = contained_file(path.parent, "episodes.json").read_bytes()
    if digest(episodes) != _sha(raw["episode_dispositions_sha256"]):
        raise StorageContractError("episode bytes differ")
    decode(episodes)
    if shared:
        batches = raw["raw_batches"]
        if not isinstance(batches, list) or len(batches) != count:
            raise StorageContractError("shared raw prefix is incomplete")
        for number, entry in enumerate(batches, 1):
            checked_bytes(root, entry)
            if entry["path"] != f"batches/batch-{number:06d}.json":
                raise StorageContractError("shared raw prefix order differs")
    return raw


def _raw_prefix(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest["schema_version"] == SHARED_SCHEMA:
        return list(manifest["raw_batches"])
    inventory = decode(contained_file(root, "inventory.json").read_bytes())
    if not isinstance(inventory, list):
        raise StorageContractError("archive inventory must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for entry in inventory:
        if not isinstance(entry, dict) or set(entry) != _FILE_FIELDS:
            raise StorageContractError("archive inventory entry differs")
        if entry["path"] in indexed:
            raise StorageContractError("duplicate archive inventory path")
        indexed[entry["path"]] = entry
    rows = []
    for number in range(1, manifest["completed_updates"] + 1):
        relative = f"batches/batch-{number:06d}.json"
        if relative not in indexed:
            raise StorageContractError("archive inventory omits required raw prefix")
        checked_bytes(root, indexed[relative])
        rows.append(indexed[relative])
    return rows


def _closure(path: Path, manifest: dict[str, Any]) -> dict[str, bytes]:
    root = archive_root(path)
    config = contained_file(root, "config.json").read_bytes()
    schedule = contained_file(root, "schedule.json").read_bytes()
    if (
        digest(config) != manifest["config_sha256"]
        or digest(schedule) != manifest["seed_schedule_sha256"]
    ):
        raise StorageContractError("archive config/schedule digest differs")
    prefix = path.parent.relative_to(root).as_posix()
    files = {
        "config.json": config,
        "schedule.json": schedule,
        f"{prefix}/manifest.json": path.read_bytes(),
        f"{prefix}/episodes.json": contained_file(path.parent, "episodes.json").read_bytes(),
    }
    for entry in manifest["artifacts"]:
        relative = (
            entry["path"]
            if manifest["schema_version"] == SHARED_SCHEMA
            else f"{prefix}/{entry['path']}"
        )
        files[relative] = checked_bytes(artifact_root(path, manifest), entry, artifact=True)
    for entry in _raw_prefix(root, manifest):
        files[entry["path"]] = checked_bytes(root, entry)
    return files


def export_checkpoint(checkpoint: str | Path, output: str | Path) -> Path:
    """Copy a single cut's complete closure, preserving original manifest bytes.

    The returned manifest uses the same nested run-relative layout. Other cuts
    are not dependencies and are not exported. Missing raw v1 lineage is a
    failure, not a silently partial export.
    """
    path = checkpoint_path(checkpoint)
    manifest = read_checkpoint(path)
    files = _closure(path, manifest)
    target = Path(output).absolute()
    target.mkdir(parents=True, exist_ok=False)
    inventory = []
    for relative, body in sorted(files.items()):
        atomic_write(target / relative, body)
        inventory.append(file_entry(relative, body))
    atomic_write(target / "inventory.json", canonical(inventory))
    export = {
        "schema_version": "local-runner-export-v1",
        "source_checkpoint_sha256": manifest["checkpoint_sha256"],
        "source_identity": manifest["source_identity"],
        "inventory_sha256": digest(canonical(inventory)),
        "storage_profile": SHARED_PROFILE
        if manifest["schema_version"] == SHARED_SCHEMA
        else STANDALONE_PROFILE,
    }
    atomic_write(target / "export.json", canonical(export))
    validate_archive(target)
    return target / path.relative_to(archive_root(path))


def seed_shared_resume(checkpoint: str | Path, output: Path) -> list[dict[str, Any]]:
    """Import a shared cut/raw prefix into an already admitted fresh run root."""
    path = checkpoint_path(checkpoint)
    manifest = read_checkpoint(path)
    if manifest["schema_version"] != SHARED_SCHEMA:
        raise StorageContractError("shared resume cannot migrate a v1 checkpoint")
    for relative, body in _closure(path, manifest).items():
        target = output / relative
        if relative in ("config.json", "schedule.json"):
            if contained_file(output, relative).read_bytes() != body:
                raise StorageContractError("resume configuration bytes differ")
        else:
            atomic_write(target, body)
    return list(manifest["raw_batches"])


def validate_archive(output: str | Path) -> dict[str, Any]:
    """Read every retained cut and its byte closure; no runtime semantic claim."""
    root = Path(output).absolute()
    _plain(root)
    for path in root.rglob("*"):
        _plain(path)
        if path.is_file() and path.name.startswith(".pending-"):
            raise StorageContractError("archive contains interrupted pending publication")
    paths = sorted((root / "checkpoints").glob("update-*/manifest.json"))
    if not paths:
        raise StorageContractError("archive contains no accepted checkpoint")
    cuts = []
    closure: dict[str, int] = {}
    profiles = set()
    for path in paths:
        manifest = read_checkpoint(path)
        if path.parent.name != f"update-{manifest['completed_updates']:06d}":
            raise StorageContractError("checkpoint directory/count differs")
        profiles.add(
            SHARED_PROFILE if manifest["schema_version"] == SHARED_SCHEMA else STANDALONE_PROFILE
        )
        cuts.append(manifest["completed_updates"])
        closure.update((relative, len(body)) for relative, body in _closure(path, manifest).items())
    if len(profiles) != 1 or len(cuts) != len(set(cuts)):
        raise StorageContractError("mixed archive profiles or duplicate cuts")
    inventory_body = contained_file(root, "inventory.json").read_bytes()
    inventory = decode(inventory_body)
    if not isinstance(inventory, list):
        raise StorageContractError("archive inventory must be a list")
    indexed = set()
    for entry in inventory:
        checked_bytes(root, entry)
        if entry["path"] in indexed:
            raise StorageContractError("duplicate archive inventory path")
        indexed.add(entry["path"])
    export_path = root / "export.json"
    if export_path.exists():
        export = decode(contained_file(root, "export.json").read_bytes())
        expected_fields = {
            "schema_version",
            "source_checkpoint_sha256",
            "source_identity",
            "inventory_sha256",
            "storage_profile",
        }
        if (
            not isinstance(export, dict)
            or set(export) != expected_fields
            or export["schema_version"] != "local-runner-export-v1"
            or export["inventory_sha256"] != digest(inventory_body)
            or len(paths) != 1
            or export["source_checkpoint_sha256"] != manifest["checkpoint_sha256"]
            or export["source_identity"] != manifest["source_identity"]
            or export["storage_profile"] not in profiles
            or set(closure) != indexed
        ):
            raise StorageContractError("export closure/provenance differs")
    return {
        "schema_version": "local-runner-archive-validation-v1",
        "byte_closure_valid": True,
        "storage_profile": next(iter(profiles)),
        "retained_cuts": cuts,
        "closure_file_count": len(closure),
        "closure_logical_bytes": sum(closure.values()),
        "semantic_equivalence_checked": False,
    }
