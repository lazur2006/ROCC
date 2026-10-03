"""Path helpers for local notebooks, scripts, and Azure jobs."""

from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path


PROJECT_ROOT_ENV = "MASTER_THESIS_PROJECT_ROOT"


def resolve_project_root() -> Path:
    env_value = os.environ.get("ROCC_ROOT") or os.environ.get(PROJECT_ROOT_ENV)
    if env_value:
        root = Path(env_value).expanduser().resolve()
        if not (root / "experiments" / "rocc").is_dir():
            raise FileNotFoundError(f"Configured ROCC checkout has no experiments/rocc: {root}")
        return root

    candidates = [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[2]]

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "experiments" / "rocc").exists():
            return candidate
        if candidate.name == "experiments" and (candidate / "rocc").exists():
            return candidate.parent

    raise FileNotFoundError("ROCC checkout not found. Set ROCC_ROOT to the directory containing experiments/rocc.")


def project_path(*parts: str | os.PathLike[str]) -> Path:
    return resolve_project_root().joinpath(*map(Path, parts))


def bind_notebook_lineage(result_dir, name, historical, files, *, manifest_fields=()):
    """Select historical identities, or explicitly pin validated local file identities.

    Only file fingerprints change in local mode. Notebook scientific assertions
    remain active. A persisted pin rejects a later change of inputs, rather than
    silently mixing results from different training runs. ``manifest_fields``
    contains (file key, manifest path, nested field names) cross-checks.
    """
    mode = os.environ.get("ROCC_LINEAGE", "historical").strip().lower()
    if mode not in {"historical", "local"}:
        raise ValueError("ROCC_LINEAGE must be historical or local.")
    if mode == "historical":
        return dict(historical)
    observed = {}
    for key, filename in files.items():
        digest = hashlib.sha256()
        with Path(filename).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        observed[key] = digest.hexdigest()
    for key, filename, fields in manifest_fields:
        value = json.loads(Path(filename).read_text(encoding="utf-8"))
        if value.get("complete") is False:
            raise ValueError(f"Incomplete upstream manifest: {filename}")
        for field in fields:
            value = value[field]
        if value != observed[key]:
            raise ValueError(f"Upstream manifest/file mismatch for {key}: {filename}")
    identity = {"mode": "local", "sha256": observed}
    pin = Path(result_dir) / "protocol" / f"{name}_local_lineage.json"
    pin.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pin.open("x", encoding="utf-8") as handle:
            json.dump(identity, handle, sort_keys=True, indent=2)
            handle.write("\n")
    except FileExistsError:
        if json.loads(pin.read_text(encoding="utf-8")) != identity:
            raise ValueError(
                f"Local input lineage changed: {pin}. Keep old results separate "
                "and use a clean result directory for a different run."
            )
    print(f"Local lineage: {pin}; historical bit-identity is not claimed.")
    return {**historical, **observed}
