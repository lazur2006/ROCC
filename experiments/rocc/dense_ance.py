"""Colab/GPU utilities for the full TopiOCQA ANCE dense index."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit
from urllib.request import urlretrieve

from .paths import project_path
from .retrievers import (
    DEFAULT_ANCE_ENCODER,
    DEFAULT_RETRIEVAL_TOP_K,
    configure_java21,
    default_dense_faiss_index_dir,
    shard_dirs,
)


TOPIOCQA_FULL_ANCE_PASSAGES = 25_700_592
ANCE_DIMENSION = 768
ANCE_QUERY_MAX_LENGTH = 64
AZCOPY_DOWNLOAD_URL = "https://aka.ms/downloadazcopy-v10-linux"
AZURE_FILE_SHARE_URL = ""  # Legacy Azure helpers require an explicit user-supplied endpoint.
TOPIOCQA_AZURE_DENSE_SHARDS_PATH = ""  # Legacy Azure helpers require an explicit user-supplied endpoint.
EncodeQueriesFn = Callable[[list[str], int], Any]


@dataclass(frozen=True)
class DenseFaissShardSummary:
    index_dir: Path
    shard_count: int
    num_passages: int
    dimension: int
    encoders: tuple[str, ...]
    max_lengths: tuple[int, ...]
    index_bytes: int
    docid_bytes: int
    contiguous: bool
    first_passage_start: int | None
    last_passage_end: int | None

    @property
    def index_gib(self) -> float:
        return self.index_bytes / 1024**3

    @property
    def docid_mib(self) -> float:
        return self.docid_bytes / 1024**2

    @property
    def fp32_gib(self) -> float:
        return self.num_passages * self.dimension * 4 / 1024**3

    @property
    def fp16_gib(self) -> float:
        return self.num_passages * self.dimension * 2 / 1024**3

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_dir": str(self.index_dir),
            "shard_count": self.shard_count,
            "num_passages": self.num_passages,
            "dimension": self.dimension,
            "encoders": self.encoders,
            "max_lengths": self.max_lengths,
            "index_gib": self.index_gib,
            "docid_mib": self.docid_mib,
            "contiguous": self.contiguous,
            "first_passage_start": self.first_passage_start,
            "last_passage_end": self.last_passage_end,
            "torch_fp32_gib": self.fp32_gib,
            "torch_fp16_gib": self.fp16_gib,
        }


@dataclass
class DenseTorchIndex:
    index_tensor: Any
    docids: list[str]
    summary: DenseFaissShardSummary


@dataclass(frozen=True)
class DenseGpuCapacity:
    required_gib: float
    free_gib: float
    total_gib: float
    fits: bool


@dataclass
class BenchmarkAnceQueryEncoder:
    tokenizer: Any
    model: Any
    device: str
    max_length: int = ANCE_QUERY_MAX_LENGTH

    def encode(self, queries: list[str], batch_size: int = 64) -> Any:
        import torch

        vectors = []
        with torch.inference_mode():
            for start in range(0, len(queries), batch_size):
                batch_queries = [str(query) for query in queries[start:start + batch_size]]
                inputs = self.tokenizer(
                    batch_queries,
                    max_length=self.max_length,
                    padding="longest",
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                vectors.append(
                    self.model(
                        inputs["input_ids"],
                        inputs.get("attention_mask"),
                    ).detach().float()
                )
        return torch.cat(vectors, dim=0)


@dataclass
class DenseAnceTorchRetriever:
    """Full ANCE retriever over a preloaded Torch tensor index.

    This is the correct path for the full TopiOCQA dense index in Colab/A100:
    FAISS shard files are used only as the storage format; retrieval itself is
    a single GPU matrix multiplication plus top-k.
    """

    index_tensor: Any
    docids: list[str]
    encode_queries: EncodeQueriesFn
    query_batch_size: int = 64

    def __post_init__(self) -> None:
        if self.index_tensor.ndim != 2:
            raise ValueError(f"index_tensor must be 2D, got {self.index_tensor.shape}")
        if len(self.docids) != int(self.index_tensor.shape[0]):
            raise ValueError(
                f"docids length {len(self.docids)} does not match index rows "
                f"{int(self.index_tensor.shape[0])}"
            )
        if self.query_batch_size < 1:
            raise ValueError("query_batch_size must be >= 1")

    @property
    def device(self) -> Any:
        return self.index_tensor.device

    @property
    def dtype(self) -> Any:
        return self.index_tensor.dtype

    def search(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        include_raw: bool = False,
    ) -> list[dict[str, Any]]:
        return self.search_batch([query], top_k=top_k, include_raw=include_raw)[0]

    def search_batch(
        self,
        queries: list[str],
        *,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        include_raw: bool = False,
    ) -> list[list[dict[str, Any]]]:
        if include_raw:
            raise NotImplementedError("DenseAnceTorchRetriever does not provide raw document text.")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not queries:
            return []

        import torch

        k = min(int(top_k), int(self.index_tensor.shape[0]))
        all_rows: list[list[dict[str, Any]]] = []
        with torch.inference_mode():
            for start in range(0, len(queries), self.query_batch_size):
                batch_queries = [str(query) for query in queries[start:start + self.query_batch_size]]
                query_vectors = self.encode_queries(batch_queries, self.query_batch_size)
                if isinstance(query_vectors, torch.Tensor):
                    query_tensor = query_vectors.to(device=self.device, dtype=self.dtype)
                else:
                    query_tensor = torch.as_tensor(query_vectors, device=self.device, dtype=self.dtype)
                scores = torch.matmul(query_tensor, self.index_tensor.T)
                values, indices = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
                values = values.detach().float().cpu().numpy()
                indices = indices.detach().cpu().numpy()
                for row_values, row_indices in zip(values, indices, strict=True):
                    all_rows.append(
                        [
                            {
                                "rank": rank,
                                "docid": str(self.docids[int(index)]),
                                "score": float(score),
                            }
                            for rank, (score, index) in enumerate(
                                zip(row_values, row_indices, strict=True),
                                start=1,
                            )
                        ]
                    )
        return all_rows

def download_dense_ance_index_from_azure(
    *,
    azure_sharded_dir_url: str | None = None,
    local_sharded_dir: Path | str | None = None,
    url_env_var: str = "AZURE_SHARDED_DIR_URL",
    url_file: Path | str | None = None,
    log_path: Path | str = "/content/azcopy_dense_index_download.log",
    install_azcopy: bool = True,
    progress: bool = True,
    include_path: str | None = None,
    from_to: str | None = None,
) -> Path:
    """Download the Azure Files ANCE shard directory with AzCopy.

    The SAS URL is intentionally never printed. Pass it directly, set
    AZURE_SHARDED_DIR_URL, or store it in a local/Drive secret file.
    """

    url = _resolve_azure_url(
        azure_sharded_dir_url=azure_sharded_dir_url,
        url_env_var=url_env_var,
        url_file=url_file,
    )
    target = Path(local_sharded_dir).expanduser().resolve() if local_sharded_dir else default_dense_faiss_index_dir()
    target.mkdir(parents=True, exist_ok=True)
    azcopy = ensure_azcopy() if install_azcopy else shutil.which("azcopy")
    if not azcopy:
        raise RuntimeError("azcopy is not available. Install it or call with install_azcopy=True.")

    log = Path(log_path).expanduser()
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(azcopy),
        "copy",
        url,
        str(target),
        "--recursive=true",
        "--overwrite=ifSourceNewer",
        "--check-length=false",
    ]
    if from_to:
        command.append(f"--from-to={from_to}")
    if include_path:
        command.append(f"--include-path={include_path}")
    if progress:
        print(f"AzCopy dense index log: {log}", flush=True)
    returncode, tail = _run_streaming_command(
        command,
        log_file=log,
        progress=progress,
        prelude=[
            f"[{_utc_now()}] start azcopy dense index download\n",
            f"target={target}\n",
        ],
    )
    if returncode != 0:
        raise RuntimeError(
            "AzCopy dense index download failed.\n"
            f"log_file: {log}\n"
            f"last_log_lines:\n{''.join(tail)}"
        )
    return _resolve_downloaded_shard_dir(target)


def default_topiocqa_ance_azure_url(
    *,
    sas_file: Path | str | None = None,
    azure_file_share_url: str = AZURE_FILE_SHARE_URL,
    azure_sharded_path: str = TOPIOCQA_AZURE_DENSE_SHARDS_PATH,
) -> str:
    resolved_sas_file = Path(sas_file).expanduser() if sas_file is not None else _default_dense_ance_sas_file()
    if not resolved_sas_file.exists():
        raise FileNotFoundError(f"Azure SAS file does not exist: {resolved_sas_file}")
    sas_query = _extract_sas_query(resolved_sas_file.read_text(encoding="utf-8").strip())
    if not sas_query:
        raise ValueError(f"Azure SAS file is empty: {resolved_sas_file}")
    return f"{azure_file_share_url.rstrip('/')}/{azure_sharded_path.strip('/')}?{sas_query}"


def ensure_azcopy(binary_path: Path | str = "/usr/local/bin/azcopy") -> Path:
    existing = shutil.which("azcopy")
    if existing:
        return Path(existing)

    binary = Path(binary_path)
    if binary.exists():
        return binary

    archive = Path("/tmp/azcopy.tar.gz")
    extract_dir = Path("/tmp/azcopy_extract")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    urlretrieve(AZCOPY_DOWNLOAD_URL, archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extract_dir)
    candidates = list(extract_dir.glob("*/azcopy")) + list(extract_dir.glob("azcopy"))
    if not candidates:
        raise RuntimeError("Downloaded AzCopy archive did not contain an azcopy binary.")
    binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[0], binary)
    binary.chmod(0o755)
    return binary


def _run_streaming_command(
    command: list[str],
    *,
    log_file: Path,
    progress: bool,
    prelude: list[str] | None = None,
    tail_lines: int = 80,
) -> tuple[int, list[str]]:
    tail: deque[str] = deque(maxlen=tail_lines)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        for line in prelude or []:
            tail.append(line)
            log.write(line)
            if progress:
                print(line, end="", flush=True)
        command_line = _redacted_command(command)
        header = f"[{_utc_now()}] command: {command_line}\n"
        tail.append(header)
        log.write(header)
        log.flush()
        if progress:
            print(header, end="", flush=True)

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("Failed to open subprocess stdout stream.")
        for line in process.stdout:
            tail.append(line)
            log.write(line)
            log.flush()
            if progress:
                print(line, end="", flush=True)
        returncode = process.wait()
        footer = f"[{_utc_now()}] exit_code: {returncode}\n"
        tail.append(footer)
        log.write(footer)
        log.flush()
        if progress:
            print(footer, end="", flush=True)
    return returncode, list(tail)


def _redacted_command(command: list[str]) -> str:
    redacted = []
    for part in command:
        if "sig=" in part or "?" in part and "://" in part:
            redacted.append(part.split("?", 1)[0] + "?<redacted-sas>")
        else:
            redacted.append(part)
    return " ".join(redacted)


def summarize_dense_faiss_shards(
    index_dir: Path | str | None = None,
    *,
    start_shard: int | None = None,
    end_shard: int | None = None,
) -> DenseFaissShardSummary:
    root = Path(index_dir).expanduser().resolve() if index_dir else default_dense_faiss_index_dir()
    root = _resolve_downloaded_shard_dir(root)
    shards = _select_shards_by_id(
        shard_dirs(root),
        start_shard=start_shard,
        end_shard=end_shard,
    )
    manifests = [_read_manifest(shard) for shard in shards]
    for shard, manifest in zip(shards, manifests, strict=True):
        if manifest.get("complete") is not True:
            raise RuntimeError(f"Incomplete ANCE shard: {shard}")
        expected_bytes = int(manifest["num_passages"]) * int(manifest["dimension"]) * 4 + 45
        if not (shard / "index").is_file() or (shard / "index").stat().st_size != expected_bytes:
            raise RuntimeError(f"Missing/truncated FAISS flat index: {shard}")
        if not (shard / "docid").is_file() or not (shard / "docid").stat().st_size:
            raise RuntimeError(f"Missing/empty ANCE docid file: {shard}")
    dimensions = {int(item["dimension"]) for item in manifests}
    if len(dimensions) != 1:
        raise RuntimeError(f"Mixed dense index dimensions: {sorted(dimensions)}")
    passage_starts = [
        int(item["passage_start"])
        for item in manifests
        if "passage_start" in item
    ]
    passage_ends = [
        int(item["passage_end"])
        for item in manifests
        if "passage_end" in item
    ]
    num_passages = sum(int(item["num_passages"]) for item in manifests)
    contiguous = True
    if len(passage_starts) == len(manifests) and len(passage_ends) == len(manifests):
        expected_start = min(passage_starts) if passage_starts else 0
        for item in sorted(manifests, key=lambda row: int(row["shard_id"])):
            start = int(item["passage_start"])
            end = int(item["passage_end"])
            if start != expected_start or end - start != int(item["num_passages"]):
                contiguous = False
                break
            expected_start = end
        contiguous = contiguous and expected_start == max(passage_ends)

    return DenseFaissShardSummary(
        index_dir=root,
        shard_count=len(shards),
        num_passages=num_passages,
        dimension=dimensions.pop(),
        encoders=tuple(sorted({str(item.get("encoder", "")) for item in manifests})),
        max_lengths=tuple(sorted({int(item["max_length"]) for item in manifests if "max_length" in item})),
        index_bytes=sum((shard / "index").stat().st_size for shard in shards),
        docid_bytes=sum((shard / "docid").stat().st_size for shard in shards),
        contiguous=contiguous,
        first_passage_start=min(passage_starts) if passage_starts else None,
        last_passage_end=max(passage_ends) if passage_ends else None,
    )


def estimate_dense_torch_index_gib(
    summary: DenseFaissShardSummary,
    *,
    index_dtype: str = "float32",
) -> float:
    if index_dtype == "float32":
        return summary.fp32_gib
    if index_dtype == "float16":
        return summary.fp16_gib
    raise ValueError("index_dtype must be 'float16' or 'float32'")


def check_dense_gpu_capacity(
    summary: DenseFaissShardSummary,
    *,
    device: str = "cuda:0",
    index_dtype: str = "float32",
    reserve_gib: float = 4.0,
) -> DenseGpuCapacity:
    import torch

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError("Full ANCE Torch retrieval requires a CUDA device.")
    gpu_id = _gpu_id(torch_device) or 0
    free, total = torch.cuda.mem_get_info(gpu_id)
    required_gib = estimate_dense_torch_index_gib(summary, index_dtype=index_dtype) + reserve_gib
    free_gib = free / 1024**3
    return DenseGpuCapacity(
        required_gib=required_gib,
        free_gib=free_gib,
        total_gib=total / 1024**3,
        fits=free_gib >= required_gib,
    )


def flat_index_memmap(index_path: Path | str, num_passages: int, dimension: int) -> tuple[Any, int]:
    import numpy as np

    path = Path(index_path)
    payload_bytes = int(num_passages) * int(dimension) * 4
    header_bytes = path.stat().st_size - payload_bytes
    if header_bytes < 0 or header_bytes > 4096:
        raise RuntimeError(f"Unexpected FAISS flat index header size: {header_bytes} bytes for {path}")
    vectors = np.memmap(
        path,
        dtype="<f4",
        mode="r",
        offset=header_bytes,
        shape=(int(num_passages), int(dimension)),
    )
    return vectors, header_bytes


def load_faiss_flat_shards_to_torch(
    index_dir: Path | str | None = None,
    *,
    device: str = "cuda:0",
    index_dtype: str = "float32",
    start_shard: int | None = None,
    end_shard: int | None = None,
    chunk_size: int = 50_000,
    progress: bool = True,
    status_log: Path | str | None = "/content/ance_index_load_status.log",
) -> DenseTorchIndex:
    if index_dtype not in {"float16", "float32"}:
        raise ValueError("index_dtype must be 'float16' or 'float32'")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    import torch
    from tqdm.auto import tqdm

    root = Path(index_dir).expanduser().resolve() if index_dir else default_dense_faiss_index_dir()
    root = _resolve_downloaded_shard_dir(root)
    shards = _select_shards_by_id(
        shard_dirs(root),
        start_shard=start_shard,
        end_shard=end_shard,
    )
    summary = summarize_dense_faiss_shards(
        root,
        start_shard=start_shard,
        end_shard=end_shard,
    )
    torch_dtype = torch.float16 if index_dtype == "float16" else torch.float32
    torch_device = torch.device(device)

    _write_load_status(status_log, "allocating dense ANCE tensor", loaded=0, gpu_id=_gpu_id(torch_device))
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    index_tensor = torch.empty(
        (summary.num_passages, summary.dimension),
        device=torch_device,
        dtype=torch_dtype,
    )
    docids: list[str] = []
    loaded = 0
    iterator = tqdm(shards, desc="load raw FAISS flat shards to torch", disable=not progress)
    with torch.inference_mode():
        for shard in iterator:
            manifest = _read_manifest(shard)
            num_passages = int(manifest["num_passages"])
            vectors, _ = flat_index_memmap(shard / "index", num_passages, summary.dimension)
            shard_docids = (shard / "docid").read_text(encoding="utf-8").splitlines()
            if len(shard_docids) != num_passages:
                raise RuntimeError(f"docid count mismatch in {shard}: {len(shard_docids)} != {num_passages}")
            for start in range(0, num_passages, chunk_size):
                end = min(start + chunk_size, num_passages)
                chunk = torch.from_numpy(vectors[start:end]).to(
                    device=torch_device,
                    dtype=torch_dtype,
                    non_blocking=torch_device.type == "cuda",
                )
                index_tensor[loaded + start:loaded + end].copy_(chunk)
            docids.extend(shard_docids)
            loaded += num_passages
            _write_load_status(
                status_log,
                f"loaded {shard.name}",
                loaded=loaded,
                gpu_id=_gpu_id(torch_device),
            )

    _write_load_status(status_log, "dense ANCE tensor ready", loaded=loaded, gpu_id=_gpu_id(torch_device))
    return DenseTorchIndex(index_tensor=index_tensor, docids=docids, summary=summary)


def load_ance_query_encoder(
    encoder_name: str = DEFAULT_ANCE_ENCODER,
    *,
    device: str = "cuda:0",
    max_length: int = ANCE_QUERY_MAX_LENGTH,
) -> BenchmarkAnceQueryEncoder:
    _prepare_pyserini_ance_imports()
    from transformers import RobertaTokenizer

    tokenizer = RobertaTokenizer.from_pretrained(encoder_name, do_lower_case=True)
    model = load_ance_encoder_exact(encoder_name).to(device).eval()
    return BenchmarkAnceQueryEncoder(
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
    )


def _local_ance_encoder_class() -> Any:
    """Return the ANCE Roberta module without importing pyserini.encode.

    Recent Pyserini versions import vision/DSE encoders from pyserini.encode's
    package __init__. In Colab this can trip broken torchvision registrations
    even though ANCE only needs this Roberta text encoder.
    """
    import torch
    from transformers import PreTrainedModel, RobertaConfig, RobertaModel, requires_backends

    class LocalAnceEncoder(PreTrainedModel):
        config_class = RobertaConfig
        base_model_prefix = "ance_encoder"
        load_tf_weights = None
        all_tied_weights_keys: dict[str, Any] = {}

        def __init__(self, config: RobertaConfig):
            requires_backends(self, "torch")
            super().__init__(config)
            self.config = config
            self.roberta = RobertaModel(config)
            self.embeddingHead = torch.nn.Linear(config.hidden_size, ANCE_DIMENSION)
            self.norm = torch.nn.LayerNorm(ANCE_DIMENSION)
            self.init_weights()

        def _init_weights(self, module: Any) -> None:
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, torch.nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        def init_weights(self) -> None:
            self.roberta.init_weights()
            self.embeddingHead.apply(self._init_weights)
            self.norm.apply(self._init_weights)

        def forward(self, input_ids: Any, attention_mask: Any | None = None) -> Any:
            input_shape = input_ids.size()
            device = input_ids.device
            if attention_mask is None:
                attention_mask = (
                    torch.ones(input_shape, device=device)
                    if input_ids is None
                    else (input_ids != self.roberta.config.pad_token_id)
                )
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0, :]
            return self.norm(self.embeddingHead(pooled_output))

    return LocalAnceEncoder


def load_ance_encoder_exact(encoder_name: str = DEFAULT_ANCE_ENCODER) -> Any:
    _prepare_pyserini_ance_imports()
    import torch
    from huggingface_hub import hf_hub_download

    AnceEncoder = _local_ance_encoder_class()
    model = AnceEncoder.from_pretrained(encoder_name)
    weights_path = hf_hub_download(encoder_name, filename="pytorch_model.bin")
    try:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_path, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    unexpected_keys = [key for key in unexpected_keys if key != "roberta.embeddings.position_ids"]
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"ANCE checkpoint load mismatch: missing={missing_keys}, unexpected={unexpected_keys}"
        )
    model.eval()
    return model


def load_dense_ance_torch_retriever(
    index_dir: Path | str | None = None,
    *,
    device: str = "cuda:0",
    index_dtype: str = "float32",
    encoder_name: str = DEFAULT_ANCE_ENCODER,
    query_encoder_device: str | None = None,
    start_shard: int | None = None,
    end_shard: int | None = None,
    chunk_size: int = 50_000,
    query_batch_size: int = 64,
    progress: bool = True,
    status_log: Path | str | None = "/content/ance_index_load_status.log",
) -> DenseAnceTorchRetriever:
    dense_index = load_faiss_flat_shards_to_torch(
        index_dir=index_dir,
        device=device,
        index_dtype=index_dtype,
        start_shard=start_shard,
        end_shard=end_shard,
        chunk_size=chunk_size,
        progress=progress,
        status_log=status_log,
    )
    query_encoder = load_ance_query_encoder(
        encoder_name=encoder_name,
        device=query_encoder_device or device,
    )
    return DenseAnceTorchRetriever(
        index_tensor=dense_index.index_tensor,
        docids=dense_index.docids,
        encode_queries=query_encoder.encode,
        query_batch_size=query_batch_size,
    )


def _select_shards_by_id(
    shards: list[Path],
    *,
    start_shard: int | None,
    end_shard: int | None,
) -> list[Path]:
    if start_shard is None and end_shard is None:
        return shards
    if start_shard is None or end_shard is None:
        raise ValueError("Pass both start_shard and end_shard, or neither.")
    if start_shard < 0 or end_shard < start_shard:
        raise ValueError(f"Invalid shard range: {start_shard}..{end_shard}")
    selected: list[Path] = []
    for shard in shards:
        manifest = _read_manifest(shard)
        shard_id = int(manifest.get("shard_id", shard.name.rsplit("_", 1)[-1]))
        if start_shard <= shard_id <= end_shard:
            selected.append(shard)
    if not selected:
        raise FileNotFoundError(
            f"No FAISS shard dirs found for shard range {start_shard}..{end_shard}"
        )
    selected_ids = sorted(
        int(_read_manifest(shard).get("shard_id", shard.name.rsplit("_", 1)[-1]))
        for shard in selected
    )
    expected_ids = list(range(start_shard, end_shard + 1))
    if selected_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(selected_ids))
        raise FileNotFoundError(
            f"Incomplete FAISS shard range {start_shard}..{end_shard}; missing {missing[:10]}"
        )
    return selected


def _resolve_azure_url(
    *,
    azure_sharded_dir_url: str | None,
    url_env_var: str,
    url_file: Path | str | None,
) -> str:
    if azure_sharded_dir_url:
        return azure_sharded_dir_url.strip()
    env_value = os.environ.get(url_env_var, "").strip()
    if env_value:
        return env_value
    if url_file:
        path = Path(url_file).expanduser()
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
    return default_topiocqa_ance_azure_url()


def _default_dense_ance_sas_file() -> Path:
    candidates = [
        Path("/content/drive/MyDrive/secrets/qrecc_file_share_rw_sas.txt"),
        Path("/content/drive/MyDrive/secrets/ance_sharded_dir_sas.txt"),
        project_path("experiments", "secrets", "qrecc_file_share_rw_sas.txt"),
        project_path("experiments", "secrets", "ance_sharded_dir_sas.txt"),
    ]
    for candidate in candidates:
        expanded = candidate.expanduser()
        if expanded.exists():
            return expanded
    return project_path("experiments", "secrets", "qrecc_file_share_rw_sas.txt")


def _extract_sas_query(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return ""
    parsed = urlsplit(stripped)
    if parsed.query:
        return parsed.query
    return stripped.lstrip("?")


def _resolve_downloaded_shard_dir(target: Path) -> Path:
    if (target / "shard_000000").exists():
        return target
    candidates = [target / "faiss_flat_index_full_sharded"]
    if target.exists():
        candidates.extend(child for child in target.iterdir() if child.is_dir())
    for candidate in candidates:
        if (candidate / "shard_000000").exists():
            return candidate
    return target


def _read_manifest(shard: Path) -> dict[str, Any]:
    manifest_path = shard / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing shard manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _prepare_pyserini_ance_imports() -> None:
    os.environ.setdefault("OPENAI_API_KEY", "pyserini-ance-faiss-no-openai-call")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    configure_java21()
    for module_name in list(sys.modules):
        if module_name == "torchvision" or module_name.startswith("torchvision."):
            sys.modules.pop(module_name, None)
    importlib.invalidate_caches()


def _write_load_status(status_log: Path | str | None, message: str, *, loaded: int, gpu_id: int | None) -> None:
    if status_log is None:
        return
    text = f"[{time.strftime('%H:%M:%S')}] {message}\nvectors_loaded: {loaded}\n"
    if gpu_id is not None:
        try:
            import torch

            free, total = torch.cuda.mem_get_info(gpu_id)
            text += f"gpu_free_gib: {free / 1024**3:.2f} / {total / 1024**3:.2f}\n"
        except Exception as exc:
            text += f"gpu_status_error: {exc!r}\n"
    path = Path(status_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _gpu_id(device: Any) -> int | None:
    if getattr(device, "type", None) != "cuda":
        return None
    return 0 if device.index is None else int(device.index)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
