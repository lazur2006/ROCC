"""Colab/Google Drive helpers for notebook runtime resources."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .paths import project_path


TOPIOCQA_CORPUS_FILENAME = "full_wiki_segments.tsv"
ITERCQR_CHECKPOINT_REQUIRED_FILES = {
    "config.json",
    "generation_config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer_config.json",
}
DEFAULT_GITHUB_REPO_SSH_URL = ""  # Supply a repository URL explicitly if cloning is wanted.


@dataclass(frozen=True)
class ColabTopiOCQACorpusResult:
    is_colab: bool
    mounted: bool
    drive_mount: Path
    corpora_dir: Path
    drive_corpus: Path | None
    local_corpus: Path
    status: str
    size_bytes: int | None

    @property
    def size_gib(self) -> float | None:
        if self.size_bytes is None:
            return None
        return self.size_bytes / 1024**3

    def to_dict(self) -> dict[str, object]:
        return {
            "is_colab": self.is_colab,
            "mounted": self.mounted,
            "drive_mount": str(self.drive_mount),
            "corpora_dir": str(self.corpora_dir),
            "drive_corpus": str(self.drive_corpus) if self.drive_corpus else None,
            "local_corpus": str(self.local_corpus),
            "status": self.status,
            "size_gib": round(self.size_gib, 2) if self.size_gib is not None else None,
        }


@dataclass(frozen=True)
class ColabIterCQRCheckpointResult:
    is_colab: bool
    mounted: bool
    drive_mount: Path
    drive_checkpoint: Path | None
    local_checkpoint: Path
    status: str
    missing_files: tuple[str, ...]
    size_bytes: int | None

    @property
    def complete(self) -> bool:
        return not self.missing_files

    @property
    def size_gib(self) -> float | None:
        if self.size_bytes is None:
            return None
        return self.size_bytes / 1024**3

    def to_dict(self) -> dict[str, object]:
        return {
            "is_colab": self.is_colab,
            "mounted": self.mounted,
            "drive_mount": str(self.drive_mount),
            "drive_checkpoint": str(self.drive_checkpoint) if self.drive_checkpoint else None,
            "local_checkpoint": str(self.local_checkpoint),
            "status": self.status,
            "complete": self.complete,
            "missing_files": list(self.missing_files),
            "size_gib": round(self.size_gib, 2) if self.size_gib is not None else None,
        }


@dataclass(frozen=True)
class ColabIterCQRRuntimeResult:
    corpus: ColabTopiOCQACorpusResult
    checkpoint: ColabIterCQRCheckpointResult


@dataclass(frozen=True)
class ColabDependencyResult:
    is_colab: bool
    installed: bool
    pyserini_available: bool
    java_home: Path | None
    java_available: bool
    packages: tuple[str, ...]
    commands_run: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "is_colab": self.is_colab,
            "installed": self.installed,
            "pyserini_available": self.pyserini_available,
            "java_home": str(self.java_home) if self.java_home else None,
            "java_available": self.java_available,
            "packages": list(self.packages),
            "commands_run": list(self.commands_run),
        }


@dataclass(frozen=True)
class ColabGitHubRepoResult:
    is_colab: bool
    mounted: bool
    repo_dir: Path
    deploy_key_source: Path
    local_deploy_key: Path
    repo_url: str
    branch: str
    cloned: bool
    pulled: bool
    commit: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "is_colab": self.is_colab,
            "mounted": self.mounted,
            "repo_dir": str(self.repo_dir),
            "deploy_key_source": str(self.deploy_key_source),
            "local_deploy_key": str(self.local_deploy_key),
            "repo_url": self.repo_url,
            "branch": self.branch,
            "cloned": self.cloned,
            "pulled": self.pulled,
            "commit": self.commit,
        }


@dataclass(frozen=True)
class ColabQReCCRuntimeResult:
    repo: ColabGitHubRepoResult | None
    dependencies: ColabDependencyResult | None

    def to_dict(self) -> dict[str, object]:
        return {
            "repo": self.repo.to_dict() if self.repo else None,
            "dependencies": self.dependencies.to_dict() if self.dependencies else None,
        }


@dataclass(frozen=True)
class NotebookBootstrapResult:
    project_root: Path
    experiments_dir: Path
    qrecc_runtime: ColabQReCCRuntimeResult | None

    def to_dict(self) -> dict[str, object]:
        dependencies = self.qrecc_runtime.dependencies if self.qrecc_runtime else None
        return {
            "project_root": str(self.project_root),
            "experiments_dir": str(self.experiments_dir),
            "is_colab": is_google_colab(),
            "dependencies_installed": dependencies.installed if dependencies else None,
            "java_home": str(dependencies.java_home) if dependencies and dependencies.java_home else None,
            "pyserini_available": dependencies.pyserini_available if dependencies else None,
        }


def is_google_colab() -> bool:
    try:
        import google.colab  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def mount_google_drive(
    mount_point: Path | str = "/content/drive",
    *,
    force_remount: bool = False,
) -> bool:
    """Mount Google Drive when running inside Colab."""

    try:
        from google.colab import drive
    except ModuleNotFoundError:
        return False
    drive.mount(str(Path(mount_point)), force_remount=force_remount)
    return True


def prepare_colab_github_repo(
    *,
    repo_dir: Path | str | None = None,
    repo_url: str = DEFAULT_GITHUB_REPO_SSH_URL,
    branch: str = "main",
    drive_deploy_key: Path | str = "",
    local_deploy_key: Path | str | None = None,
    drive_mount: Path | str = "/content/drive",
    force_remount: bool = False,
    pull: bool = False,
    allow_local: bool = False,
) -> ColabGitHubRepoResult:
    """Use a prepared checkout; optionally clone an explicitly supplied URL.

    Public HTTPS and already-configured private Git authentication both work.
    Legacy credential arguments remain accepted, but no keys, SSH configuration,
    Git remotes or global Git settings are modified. Drive mounting is separate.
    """
    target_repo = Path(repo_dir).expanduser().resolve() if repo_dir else project_path()
    cloned = False
    if not (target_repo / "experiments" / "rocc").is_dir():
        if not repo_url:
            raise FileNotFoundError(
                f"Prepare a ROCC checkout at {target_repo}, then set ROCC_ROOT. "
                "Clone public repositories over HTTPS or use your own private Git authentication."
            )
        if target_repo.exists() and any(target_repo.iterdir()):
            raise RuntimeError(f"Refusing to clone into nonempty directory: {target_repo}")
        _run_checked(["git", "clone", "--branch", branch, repo_url, str(target_repo)])
        cloned = True
    pulled = False
    if pull:
        _run_checked(["git", "-C", str(target_repo), "pull", "--ff-only", "origin", branch])
        pulled = True
    commit = None
    if (target_repo / ".git").exists():
        commit = subprocess.check_output(
            ["git", "-C", str(target_repo), "rev-parse", "--short", "HEAD"], text=True,
        ).strip()
    return ColabGitHubRepoResult(
        is_colab=is_google_colab(), mounted=False, repo_dir=target_repo,
        deploy_key_source=Path(drive_deploy_key),
        local_deploy_key=Path(local_deploy_key or "."),
        repo_url=repo_url, branch=branch, cloned=cloned, pulled=pulled, commit=commit,
    )


def prepare_colab_qrecc_runtime(
    *,
    repo_dir: Path | str | None = None,
    sync_repo: bool | None = None,
    repo_url: str = DEFAULT_GITHUB_REPO_SSH_URL,
    branch: str = "main",
    drive_deploy_key: Path | str = "",
    pyserini_version: str = "2.0.0",
) -> ColabQReCCRuntimeResult:
    """Prepare the Colab runtime needed by the QReCC notebook."""

    running_in_colab = is_google_colab()
    should_sync = False if sync_repo is None else sync_repo
    repo = (
        prepare_colab_github_repo(
            repo_dir=repo_dir,
            repo_url=repo_url,
            branch=branch,
            drive_deploy_key=drive_deploy_key,
        )
        if should_sync
        else None
    )
    dependencies = ensure_colab_retrieval_dependencies(pyserini_version=pyserini_version)
    return ColabQReCCRuntimeResult(repo=repo, dependencies=dependencies)


def bootstrap_notebook(
    *,
    prepare_qrecc_runtime: bool = True,
    sync_repo: bool = False,
    pyserini_version: str = "2.0.0",
) -> NotebookBootstrapResult:
    """Finish lightweight notebook setup after the repo is already importable."""

    root = project_path()
    experiments_dir = root / "experiments"
    if str(experiments_dir) not in sys.path:
        sys.path.insert(0, str(experiments_dir))

    qrecc_runtime = (
        prepare_colab_qrecc_runtime(
            sync_repo=sync_repo,
            pyserini_version=pyserini_version,
        )
        if prepare_qrecc_runtime
        else None
    )
    return NotebookBootstrapResult(
        project_root=root,
        experiments_dir=experiments_dir,
        qrecc_runtime=qrecc_runtime,
    )


def default_topiocqa_corpus_path() -> Path:
    return project_path(
        "experiments",
        "data",
        "topiocqa",
        "downloads",
        "data",
        "wikipedia_split",
        TOPIOCQA_CORPUS_FILENAME,
    )


def default_itercqr_checkpoint_path() -> Path:
    return project_path("experiments", "model", "IterCQR", "IterCQR Model")


def find_drive_corpus(
    *,
    corpora_dir: Path | str = "/content/drive/MyDrive/corpora",
    filename: str = TOPIOCQA_CORPUS_FILENAME,
    candidates: Iterable[Path | str] | None = None,
) -> Path | None:
    """Find a corpus file in Drive, preferring common stable paths."""

    root = Path(corpora_dir)
    known = [
        root / filename,
        root / "topiocqa" / filename,
    ]
    if candidates is not None:
        known.extend(Path(path) for path in candidates)
    for path in known:
        if path.exists():
            return path
    if root.exists():
        matches = sorted(root.rglob(filename))
        if matches:
            return matches[0]
    return None


def find_drive_itercqr_checkpoint(
    *,
    drive_mount: Path | str = "/content/drive",
    candidates: Iterable[Path | str] | None = None,
) -> Path | None:
    """Find the IterCQR checkpoint directory in Google Drive."""

    mount = Path(drive_mount)
    my_drive = mount / "MyDrive"
    known = [
        my_drive / "IterCQR" / "IterCQR Model",
        my_drive / "IterCQR",
        my_drive / "models" / "IterCQR" / "IterCQR Model",
        my_drive / "ROCC" / "experiments" / "model" / "IterCQR" / "IterCQR Model",
    ]
    if candidates is not None:
        known.extend(Path(path) for path in candidates)
    for path in known:
        if _checkpoint_missing_files(path) == ():
            return path
    root = my_drive / "IterCQR"
    if root.exists():
        for path in sorted([root, *root.rglob("*")]):
            if path.is_dir() and _checkpoint_missing_files(path) == ():
                return path
    return None


def prepare_colab_topiocqa_corpus(
    *,
    drive_mount: Path | str = "/content/drive",
    corpora_dir: Path | str | None = None,
    local_corpus: Path | str | None = None,
    force_remount: bool = False,
    replace_broken_symlink: bool = True,
) -> ColabTopiOCQACorpusResult:
    """Mount Drive and link the TopiOCQA wiki corpus into the local data tree."""

    mount = Path(drive_mount)
    mounted = False
    root = Path(corpora_dir) if corpora_dir else mount / "MyDrive" / "corpora"
    target = Path(local_corpus).expanduser().resolve() if local_corpus else default_topiocqa_corpus_path()

    if target.exists():
        return ColabTopiOCQACorpusResult(
            is_colab=is_google_colab(),
            mounted=mounted,
            drive_mount=mount,
            corpora_dir=root,
            drive_corpus=None,
            local_corpus=target,
            status="local corpus already exists",
            size_bytes=target.stat().st_size,
        )

    mounted = mount_google_drive(mount, force_remount=force_remount)

    if target.is_symlink() and not target.exists() and replace_broken_symlink:
        target.unlink()

    drive_corpus = find_drive_corpus(corpora_dir=root)
    if drive_corpus is None:
        return ColabTopiOCQACorpusResult(
            is_colab=is_google_colab(),
            mounted=mounted,
            drive_mount=mount,
            corpora_dir=root,
            drive_corpus=None,
            local_corpus=target,
            status="drive corpus not found",
            size_bytes=None,
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to(drive_corpus)
        status = "linked"
    elif target.is_symlink() and target.resolve() == drive_corpus.resolve():
        status = "already linked"
    else:
        status = "local corpus already exists"

    return ColabTopiOCQACorpusResult(
        is_colab=is_google_colab(),
        mounted=mounted,
        drive_mount=mount,
        corpora_dir=root,
        drive_corpus=drive_corpus,
        local_corpus=target,
        status=status,
        size_bytes=drive_corpus.stat().st_size,
    )


def prepare_colab_itercqr_checkpoint(
    *,
    drive_mount: Path | str = "/content/drive",
    local_checkpoint: Path | str | None = None,
    force_remount: bool = False,
    replace_broken_symlink: bool = True,
) -> ColabIterCQRCheckpointResult:
    """Mount Drive and link the IterCQR checkpoint into the local model tree."""

    mount = Path(drive_mount)
    mounted = False
    target = Path(local_checkpoint).expanduser().resolve() if local_checkpoint else default_itercqr_checkpoint_path()

    if target.exists():
        missing = _checkpoint_missing_files(target)
        return ColabIterCQRCheckpointResult(
            is_colab=is_google_colab(),
            mounted=mounted,
            drive_mount=mount,
            drive_checkpoint=None,
            local_checkpoint=target,
            status="local checkpoint already exists",
            missing_files=missing,
            size_bytes=_directory_size(target) if not missing else None,
        )

    mounted = mount_google_drive(mount, force_remount=force_remount)

    if target.is_symlink() and not target.exists() and replace_broken_symlink:
        target.unlink()

    drive_checkpoint = find_drive_itercqr_checkpoint(drive_mount=mount)
    if drive_checkpoint is None:
        return ColabIterCQRCheckpointResult(
            is_colab=is_google_colab(),
            mounted=mounted,
            drive_mount=mount,
            drive_checkpoint=None,
            local_checkpoint=target,
            status="drive checkpoint not found",
            missing_files=tuple(sorted(ITERCQR_CHECKPOINT_REQUIRED_FILES)),
            size_bytes=None,
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to(drive_checkpoint, target_is_directory=True)
        status = "linked"
    elif target.is_symlink() and target.resolve() == drive_checkpoint.resolve():
        status = "already linked"
    else:
        status = "local checkpoint already exists"

    missing = _checkpoint_missing_files(target)
    return ColabIterCQRCheckpointResult(
        is_colab=is_google_colab(),
        mounted=mounted,
        drive_mount=mount,
        drive_checkpoint=drive_checkpoint,
        local_checkpoint=target,
        status=status,
        missing_files=missing,
        size_bytes=_directory_size(drive_checkpoint) if not missing else None,
    )


def prepare_colab_itercqr_runtime(
    *,
    drive_mount: Path | str = "/content/drive",
    corpora_dir: Path | str | None = None,
    force_remount: bool = False,
) -> ColabIterCQRRuntimeResult:
    """Prepare Drive-backed TopiOCQA corpus and IterCQR checkpoint for notebooks."""

    corpus = prepare_colab_topiocqa_corpus(
        drive_mount=drive_mount,
        corpora_dir=corpora_dir,
        force_remount=force_remount,
    )
    checkpoint = prepare_colab_itercqr_checkpoint(
        drive_mount=drive_mount,
        force_remount=False,
    )
    return ColabIterCQRRuntimeResult(corpus=corpus, checkpoint=checkpoint)


def ensure_colab_retrieval_dependencies(
    *,
    pyserini_version: str = "2.0.0",
    install: bool | None = None,
) -> ColabDependencyResult:
    """Ensure Colab has the runtime deps needed for BM25/Lucene and ANCE encoders."""

    running_in_colab = is_google_colab()
    should_install = running_in_colab if install is None else install
    commands_run: list[str] = []
    packages = (f"pyserini=={pyserini_version}",)

    java_home = _find_java21_home()
    pyserini_available = importlib.util.find_spec("pyserini") is not None
    installed = False

    if should_install and java_home is None:
        _run_checked(["apt-get", "-qq", "update"])
        _run_checked(["apt-get", "-qq", "install", "-y", "openjdk-21-jdk-headless"])
        commands_run.extend(
            [
                "apt-get -qq update",
                "apt-get -qq install -y openjdk-21-jdk-headless",
            ]
        )
        installed = True
        java_home = _find_java21_home()

    if should_install and not pyserini_available:
        _run_checked(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--upgrade",
                "--no-cache-dir",
                f"pyserini=={pyserini_version}",
            ]
        )
        commands_run.append(f"{sys.executable} -m pip install pyserini=={pyserini_version}")
        installed = True
        importlib.invalidate_caches()
        pyserini_available = importlib.util.find_spec("pyserini") is not None

    java_home = _find_java21_home()
    if java_home is not None:
        _configure_java(java_home)

    return ColabDependencyResult(
        is_colab=running_in_colab,
        installed=installed,
        pyserini_available=pyserini_available,
        java_home=java_home,
        java_available=java_home is not None,
        packages=packages,
        commands_run=tuple(commands_run),
    )


def _checkpoint_missing_files(path: Path) -> tuple[str, ...]:
    if not path.exists() or not path.is_dir():
        return tuple(sorted(ITERCQR_CHECKPOINT_REQUIRED_FILES))
    existing = {child.name for child in path.iterdir() if child.is_file()}
    return tuple(sorted(ITERCQR_CHECKPOINT_REQUIRED_FILES - existing))


def _directory_size(path: Path) -> int:
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _find_java21_home() -> Path | None:
    candidates = [
        Path("/usr/lib/jvm/java-21-openjdk-amd64"),
    ]
    if Path("/usr/lib/jvm").exists():
        candidates.extend(sorted(Path("/usr/lib/jvm").glob("java-21*")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _configure_java(java_home: Path) -> None:
    os.environ["JAVA_HOME"] = str(java_home)
    os.environ["PATH"] = f"{java_home / 'bin'}:{os.environ['PATH']}"
    jvm_path = java_home / "lib" / "server" / "libjvm.so"
    if jvm_path.exists():
        os.environ["JVM_PATH"] = str(jvm_path)


def _run_checked(command: list[str], *, allow_return_codes: set[int] | None = None) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    allowed = allow_return_codes or {0}
    if result.returncode not in allowed:
        raise RuntimeError(
            "Command failed while preparing Colab dependencies.\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
