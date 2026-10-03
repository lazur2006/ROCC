"""Progress and download utilities that render well in notebooks."""

from __future__ import annotations

import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, TypeVar


T = TypeVar("T")
USER_AGENT = "rocc-experiments/1.0"


def get_tqdm():
    try:
        from tqdm.auto import tqdm

        return tqdm
    except ModuleNotFoundError:
        return _plain_tqdm


def progress_iter(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str | None = None,
    unit: str | None = None,
    enabled: bool = True,
) -> Iterable[T]:
    if not enabled:
        return iterable
    return get_tqdm()(iterable, total=total, desc=desc, unit=unit)


def download_file(
    url: str,
    target: Path,
    *,
    desc: str | None = None,
    force: bool = False,
    timeout: float = 60.0,
    chunk_size: int = 1024 * 1024,
    progress: bool = True,
) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        return target

    tmp_target = target.with_suffix(target.suffix + ".part")
    if tmp_target.exists():
        tmp_target.unlink()

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header else None
        bar = get_tqdm()(
            total=total,
            unit="B",
            unit_scale=True,
            desc=desc or target.name,
            disable=not progress,
        )
        try:
            with tmp_target.open("wb") as handle:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
                    bar.update(len(chunk))
        finally:
            bar.close()

    tmp_target.replace(target)
    elapsed = max(time.time() - started, 0.001)
    if progress:
        print(f"downloaded {target} ({format_bytes(target.stat().st_size)}, {elapsed:.1f}s)")
    return target


def copy_file(
    source: Path | str,
    target: Path | str,
    *,
    desc: str | None = None,
    force: bool = False,
    chunk_size: int = 64 * 1024 * 1024,
    progress: bool = True,
) -> Path:
    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not force and target.stat().st_size == source.stat().st_size:
        return target

    tmp_target = target.with_suffix(target.suffix + ".part")
    if tmp_target.exists():
        tmp_target.unlink()

    total = source.stat().st_size
    bar = get_tqdm()(
        total=total,
        unit="B",
        unit_scale=True,
        desc=desc or f"copy {source.name}",
        disable=not progress,
    )
    try:
        with source.open("rb") as source_handle, tmp_target.open("wb") as target_handle:
            while True:
                chunk = source_handle.read(chunk_size)
                if not chunk:
                    break
                target_handle.write(chunk)
                bar.update(len(chunk))
    finally:
        bar.close()

    tmp_target.replace(target)
    if progress:
        print(f"copied {source} -> {target} ({format_bytes(target.stat().st_size)})")
    return target


def extract_zip(
    zip_path: Path | str,
    target_dir: Path | str,
    *,
    desc: str | None = None,
    force: bool = False,
    chunk_size: int = 8 * 1024 * 1024,
    progress: bool = True,
) -> Path:
    zip_path = Path(zip_path)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    marker = target_dir / ".rocc_extract_complete"

    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        file_members = [member for member in members if not member.is_dir()]
        if marker.exists() and not force:
            return target_dir

        if file_members and not force:
            expected = [
                (target_dir / member.filename, member.file_size)
                for member in file_members
            ]
            if all(path.exists() and path.stat().st_size == size for path, size in expected):
                marker.write_text(zip_path.name + "\n", encoding="utf-8")
                return target_dir

        total = sum(member.file_size for member in file_members)
        bar = get_tqdm()(
            total=total,
            unit="B",
            unit_scale=True,
            desc=desc or f"unzip {zip_path.name}",
            disable=not progress,
        )
        try:
            for member in members:
                _validate_zip_member(target_dir, member.filename)
                _extract_zip_member(
                    archive,
                    member,
                    target_dir,
                    bar,
                    chunk_size=chunk_size,
                )
        finally:
            bar.close()

    marker.write_text(zip_path.name + "\n", encoding="utf-8")
    if progress:
        print(f"extracted {zip_path} -> {target_dir}")
    return target_dir


def format_bytes(size: int | None) -> str:
    if size is None:
        return "unknown"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def _plain_tqdm(iterable=None, **kwargs):
    if iterable is None:
        return _PlainProgress(**kwargs)
    return iterable


class _PlainProgress:
    def __init__(self, **kwargs) -> None:
        self.total = kwargs.get("total")
        self.n = 0

    def update(self, value: int = 1) -> None:
        self.n += value

    def close(self) -> None:
        return None


def _validate_zip_member(target_dir: Path, member_name: str) -> None:
    target_root = target_dir.resolve()
    member_path = (target_dir / member_name).resolve()
    if target_root != member_path and target_root not in member_path.parents:
        raise ValueError(f"Unsafe zip member path: {member_name}")


def _extract_zip_member(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    target_dir: Path,
    bar: object,
    *,
    chunk_size: int,
) -> None:
    target = target_dir / member.filename
    if member.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as source_handle, target.open("wb") as target_handle:
        while True:
            chunk = source_handle.read(chunk_size)
            if not chunk:
                break
            target_handle.write(chunk)
            bar.update(len(chunk))
