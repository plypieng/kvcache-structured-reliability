from __future__ import annotations

import argparse
import codecs
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import socket
import subprocess
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TypeVar

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from toy_kv_experiments import real_quantized_cache
from toy_kv_experiments.real_quantized_cache import RealQuantizedCache, RealQuantizedCacheConfig


DEFAULT_MODEL_DIR = Path("toy_kv_experiments/models/qwen2_5_0_5b_instruct")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
T = TypeVar("T")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Write JSON through an adjacent temporary file and atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _optional_package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _small_file_sha256(path: Path) -> str | None:
    return _file_sha256(path) if path.is_file() else None


def _model_file_manifest(model_dir: Path) -> list[dict[str, Any]]:
    patterns = ("*.safetensors", "*.bin", "*.index.json", "config.json", "tokenizer*.json")
    paths = sorted({path for pattern in patterns for path in model_dir.glob(pattern) if path.is_file()})
    return [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    ]


def build_run_metadata(
    *,
    model_dir: str | Path,
    model_name: str,
    device: str,
    protocol: dict[str, Any],
    generation_seed: int,
) -> dict[str, Any]:
    """Capture enough provenance to identify code, model, protocol, and hardware."""
    model_path = Path(model_dir).expanduser().resolve()
    source_paths = [Path(__file__).resolve(), Path(real_quantized_cache.__file__).resolve()]
    source_hashes = {str(path.relative_to(PROJECT_ROOT)): _file_sha256(path) for path in source_paths}
    combined_source_hash = hashlib.sha256(
        "\n".join(f"{name}:{digest}" for name, digest in sorted(source_hashes.items())).encode("utf-8")
    ).hexdigest()
    structeval_root = PROJECT_ROOT / "third_party" / "StructEval"
    device_name = device
    if device == "cuda" and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    elif device == "mps":
        device_name = "Apple Metal Performance Shaders"
    metadata: dict[str, Any] = {
        "run_id": uuid.uuid4().hex,
        "run_started_at_utc": utc_now_iso(),
        "protocol_label": "deterministic-structeval-t-json-subset",
        "protocol": protocol,
        "generation_seed": int(generation_seed),
        "command": sys.argv,
        "project_git_commit": _git_revision(PROJECT_ROOT),
        "structeval_git_commit": _git_revision(structeval_root),
        "source_sha256": source_hashes,
        "combined_source_sha256": combined_source_hash,
        "model": {
            "declared_name": model_name or None,
            "local_path": str(model_path),
            "config_sha256": _small_file_sha256(model_path / "config.json"),
            "tokenizer_config_sha256": _small_file_sha256(model_path / "tokenizer_config.json"),
            "files": _model_file_manifest(model_path),
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _optional_package_version("transformers"),
            "huggingface_hub": _optional_package_version("huggingface-hub"),
            "safetensors": _optional_package_version("safetensors"),
            "device": device,
            "device_name": device_name,
            "cuda_version": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic) if torch.backends.cudnn.is_available() else None,
        },
    }
    fingerprint_payload = {
        "protocol": metadata["protocol"],
        "generation_seed": metadata["generation_seed"],
        "combined_source_sha256": combined_source_hash,
        "model": metadata["model"],
        "structeval_git_commit": metadata["structeval_git_commit"],
    }
    metadata["run_fingerprint"] = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return metadata


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_dtype_for_device(device: str) -> torch.dtype:
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def resolve_model_dir(model_dir: str | Path = DEFAULT_MODEL_DIR) -> Path:
    """Resolve model path robustly from scripts or notebooks.

    In notebooks, the current working directory may be `notebooks/`. If a
    relative model path is not found from the current directory, try resolving
    it from the project root before giving it to Hugging Face Transformers.
    """
    model_dir = Path(model_dir).expanduser()
    if model_dir.exists():
        return model_dir.resolve()

    if not model_dir.is_absolute():
        project_relative = PROJECT_ROOT / model_dir
        if project_relative.exists():
            return project_relative.resolve()

    raise FileNotFoundError(
        f"local model directory not found: {model_dir}\n"
        f"Try: hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir {PROJECT_ROOT / DEFAULT_MODEL_DIR}"
    )


def load_qwen(model_dir: str | Path = DEFAULT_MODEL_DIR, device: str | None = None):
    device = device or pick_device()
    model_dir = resolve_model_dir(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=model_dtype_for_device(device),
    ).to(device)
    model.eval()
    return model, tokenizer, device


def fake_quantize_symmetric(
    x: torch.Tensor,
    bits: int,
    dim: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Quantize -> dequantize simulation.

    This keeps tensors as floating point, so it does not save real memory.
    It injects approximately the same value error that low-bit storage would
    introduce, which is enough for learning and first-pass evaluation.
    """
    if bits >= 16:
        return x
    if bits < 2:
        raise ValueError("bits must be >= 2")

    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=dim, keepdim=True).clamp_min(eps) / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return q * scale


def fake_quantize_cache_tensor(
    x: torch.Tensor,
    bits: int,
    scale_dim: int,
    residual_length: int = 0,
    protected_positions: torch.Tensor | None = None,
    protected_bits: int | None = None,
    token_dim: int = -2,
) -> torch.Tensor:
    """Fake-quantize a cache tensor while preserving recent token positions.

    KIVI-style implementations keep a recent residual cache in higher precision.
    Our first fidelity-allocation experiment also allows arbitrary token positions
    to receive higher fidelity after base quantization.
    For our educational fake-quantization experiment, this means:
      - most token positions are quantized/dequantized,
      - protected positions are copied unchanged or quantized with `protected_bits`.
    """
    if bits >= 16:
        return x

    token_dim = token_dim if token_dim >= 0 else x.ndim + token_dim
    seq_len = x.shape[token_dim]

    residual_mask = torch.zeros(seq_len, dtype=torch.bool, device=x.device)
    protected_mask = torch.zeros(seq_len, dtype=torch.bool, device=x.device)
    if residual_length > 0:
        if residual_length >= seq_len:
            return x
        residual_mask[-residual_length:] = True
    if protected_positions is not None:
        positions = protected_positions.to(device=x.device, dtype=torch.bool).flatten()
        if positions.numel() >= seq_len:
            positions = positions[-seq_len:]
        else:
            pad = torch.zeros(seq_len - positions.numel(), dtype=torch.bool, device=x.device)
            positions = torch.cat([pad, positions], dim=0)
        protected_mask |= positions

    quantized = fake_quantize_symmetric(x, bits=bits, dim=scale_dim)
    if not bool(residual_mask.any() or protected_mask.any()):
        return quantized

    out = quantized
    view_shape = [1] * x.ndim
    view_shape[token_dim] = seq_len

    if protected_bits is None or protected_bits >= 16:
        protected_values = x
    else:
        protected_values = fake_quantize_symmetric(x, bits=protected_bits, dim=scale_dim)
    if bool(protected_mask.any()):
        out = torch.where(protected_mask.view(view_shape), protected_values, out)

    if bool(residual_mask.any()):
        out = torch.where(residual_mask.view(view_shape), x, out)
    return out


def fake_quantize_cache_tensor_blockwise(
    x: torch.Tensor,
    bits: int,
    scale_dim: int,
    quantized_prefix_length: int,
    residual_length: int,
    block_size: int = 1,
    protected_positions: torch.Tensor | None = None,
    protected_bits: int | None = None,
    token_dim: int = -2,
) -> tuple[torch.Tensor, int]:
    """Quantize only newly evicted cache positions.

    Positions before `quantized_prefix_length` are assumed to be already
    quantized and are left unchanged. The newest `residual_length` positions are
    kept in full precision. Only complete evicted blocks are quantized.
    """
    if bits >= 16:
        return x, x.shape[token_dim]

    token_dim = token_dim if token_dim >= 0 else x.ndim + token_dim
    seq_len = x.shape[token_dim]
    quantized_prefix_length = max(0, min(int(quantized_prefix_length), seq_len))
    residual_length = max(0, int(residual_length))
    block_size = max(1, int(block_size))

    target_prefix = max(0, seq_len - residual_length)
    evictable = target_prefix - quantized_prefix_length
    if evictable < block_size:
        return x, quantized_prefix_length
    target_prefix = quantized_prefix_length + (evictable // block_size) * block_size
    if target_prefix <= quantized_prefix_length:
        return x, quantized_prefix_length

    quant_slice = [slice(None)] * x.ndim
    quant_slice[token_dim] = slice(quantized_prefix_length, target_prefix)
    quant_slice_tuple = tuple(quant_slice)

    slice_protection = None
    if protected_positions is not None:
        positions = protected_positions.to(device=x.device, dtype=torch.bool).flatten()
        if positions.numel() >= seq_len:
            positions = positions[-seq_len:]
        else:
            pad = torch.zeros(seq_len - positions.numel(), dtype=torch.bool, device=x.device)
            positions = torch.cat([pad, positions], dim=0)
        slice_protection = positions[quantized_prefix_length:target_prefix]

    quantized_slice = fake_quantize_cache_tensor(
        x[quant_slice_tuple],
        bits=bits,
        scale_dim=scale_dim,
        residual_length=0,
        protected_positions=slice_protection,
        protected_bits=protected_bits,
        token_dim=token_dim,
    )
    out = x.clone()
    out[quant_slice_tuple] = quantized_slice
    return out, target_prefix


JSON_STRUCTURE_MARKERS = frozenset({"{", "}", "[", "]", ":", ",", '"'})
STRUCTEVAL_CONTROL_MARKERS = ("<|BEGIN_CODE|>", "<|END_CODE|>", "BEGIN_CODE", "END_CODE")
STRUCTURE_TOKEN_PROTECTION_MODES = (
    "none",
    "all",
    "json-syntax",
    "constraint-paths",
    "json-syntax+constraint-paths",
)
PROTECTION_BUDGET_ORDERS = ("prefix", "recent", "random", "score")
PROTECTION_TARGETS = ("both", "keys", "values")
PROTECTION_SIGNAL_SOURCES = ("prompt-visible", "oracle-required-paths")


def _normalized_identifier_parts(text: str) -> list[str]:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_").lower()
    if not normalized:
        return []
    parts = [part for part in normalized.split("_") if len(part) >= 2]
    if len(normalized) >= 2:
        parts.append(normalized)
    return parts


def constraint_terms_from_required_paths(required_items: list[Any]) -> list[str]:
    """Extract schema/path terms from StructEval's required JSON paths.

    Example: `novel.author.birth_year` protects `novel`, `author`,
    `birth_year`, `birth`, and `year`. This targets prompt-side schema memory,
    which is the part delimiter-only protection missed.
    """
    terms: set[str] = set()
    for item in required_items:
        for part in path_parts(str(item)):
            if not isinstance(part, str) or part == "*":
                continue
            terms.update(_normalized_identifier_parts(part))
    return sorted(terms)


def prompt_visible_constraint_terms(prompt: str) -> list[str]:
    """Extract field-like identifiers only from text the model receives.

    StructEval conversion prompts include source code, so XML tags, TOML keys,
    JSON keys, and CSV headers are deployable signals. This intentionally does
    not inspect `raw_output_metric`, which is evaluator-side information.
    """
    candidates: set[str] = set()
    code_sections = re.findall(r"<code>(.*?)</code>", prompt, flags=re.DOTALL | re.IGNORECASE) or [prompt]
    for section in code_sections:
        candidates.update(re.findall(r"<\/?([A-Za-z_][A-Za-z0-9_-]*)\b", section))
        candidates.update(re.findall(r'"([A-Za-z_][A-Za-z0-9_.-]*)"\s*:', section))
        candidates.update(
            re.findall(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_.-]*(?:\[[0-9*]+\])?)\s*=", section)
        )
        candidates.update(re.findall(r"(?m)^\s*\[\[?\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*\]\]?", section))
        for line in section.splitlines():
            stripped = line.strip()
            if "," not in stripped or " " in stripped:
                continue
            fields = [field.strip() for field in stripped.split(",")]
            if all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.\[\]-]*", field or "") for field in fields):
                candidates.update(fields)
    terms: set[str] = set()
    for candidate in candidates:
        terms.update(_normalized_identifier_parts(candidate))
    return sorted(terms)


def _add_identifier_weight(weights: dict[str, float], text: str, score: float) -> None:
    terms = _normalized_identifier_parts(text)
    if not terms:
        return
    full_term = terms[-1]
    weights[full_term] = max(weights.get(full_term, 0.0), score)
    for term in terms[:-1]:
        weights[term] = max(weights.get(term, 0.0), score * 0.5)


def constraint_term_weights_from_required_paths(required_items: list[Any]) -> dict[str, float]:
    """Assign larger weights to deeper and leaf schema/path terms.

    This is a first-pass structural failure-cost proxy: losing a leaf field
    such as `birth_year` usually breaks a required path more directly than
    slightly damaging a broad parent key such as `novel`.
    """
    weights: dict[str, float] = {}
    for item in required_items:
        keys = [part for part in path_parts(str(item)) if isinstance(part, str) and part != "*"]
        if not keys:
            continue
        for index, key in enumerate(keys):
            score = float(index + 1)
            if index == len(keys) - 1:
                score += float(len(keys))
            _add_identifier_weight(weights, key, score)
    return weights


def constraint_term_weights_from_terms(terms: list[str]) -> dict[str, float]:
    """Give prompt-visible identifiers a stable, non-oracle importance weight."""
    weights: dict[str, float] = {}
    for term in terms:
        _add_identifier_weight(weights, term, 1.0)
    return weights


def resolve_protection_signals(
    prompt: str,
    required_items: list[Any],
    source: str,
) -> tuple[list[str], dict[str, float]]:
    if source not in PROTECTION_SIGNAL_SOURCES:
        raise ValueError(f"unknown protection signal source: {source}")
    if source == "oracle-required-paths":
        return (
            constraint_terms_from_required_paths(required_items),
            constraint_term_weights_from_required_paths(required_items),
        )
    terms = prompt_visible_constraint_terms(prompt)
    return terms, constraint_term_weights_from_terms(terms)


def _piece_matches_constraint_terms(piece: str, constraint_terms: set[str]) -> bool:
    if not constraint_terms:
        return False
    piece_terms = set(_normalized_identifier_parts(piece))
    if not piece_terms:
        return False
    for piece_term in piece_terms:
        if piece_term in constraint_terms:
            return True
        if any(piece_term in term or term in piece_term for term in constraint_terms):
            return True
    return False


def _piece_constraint_score(piece: str, constraint_term_weights: dict[str, float]) -> float:
    if not constraint_term_weights:
        return 0.0
    piece_terms = set(_normalized_identifier_parts(piece))
    if not piece_terms:
        return 0.0
    best_score = 0.0
    for piece_term in piece_terms:
        for term, score in constraint_term_weights.items():
            if piece_term == term or piece_term in term or term in piece_term:
                best_score = max(best_score, float(score))
    return best_score


def structure_token_mask(
    tokenizer,
    token_ids: torch.Tensor,
    mode: str = "none",
    constraint_terms: list[str] | None = None,
) -> torch.Tensor | None:
    """Return a boolean mask for token positions that should keep higher fidelity."""
    if mode == "none":
        return None
    if mode not in STRUCTURE_TOKEN_PROTECTION_MODES:
        raise ValueError(f"unknown structure token protection mode: {mode}")

    ids = token_ids.detach().cpu().flatten().tolist()
    normalized_constraint_terms = {
        term
        for raw_term in (constraint_terms or [])
        for term in _normalized_identifier_parts(str(raw_term))
    }
    protected: list[bool] = []
    for token_id in ids:
        piece = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        is_control = any(marker in piece for marker in STRUCTEVAL_CONTROL_MARKERS)
        is_json_syntax = any(marker in piece for marker in JSON_STRUCTURE_MARKERS)
        is_constraint = _piece_matches_constraint_terms(piece, normalized_constraint_terms)
        if mode == "all":
            protected.append(True)
        elif mode == "json-syntax":
            protected.append(is_control or is_json_syntax)
        elif mode == "constraint-paths":
            protected.append(is_control or is_constraint)
        else:
            protected.append(is_control or is_json_syntax or is_constraint)
    return torch.tensor(protected, dtype=torch.bool, device=token_ids.device)


def structure_token_scores(
    tokenizer,
    token_ids: torch.Tensor,
    mode: str = "none",
    constraint_term_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Return per-position structural importance scores for budgeted UEP."""
    if mode == "none":
        return torch.zeros_like(token_ids, dtype=torch.float32)
    if mode not in STRUCTURE_TOKEN_PROTECTION_MODES:
        raise ValueError(f"unknown structure token protection mode: {mode}")

    ids = token_ids.detach().cpu().flatten().tolist()
    weights = constraint_term_weights or {}
    scores: list[float] = []
    for token_id in ids:
        piece = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        is_control = any(marker in piece for marker in STRUCTEVAL_CONTROL_MARKERS)
        is_json_syntax = any(marker in piece for marker in JSON_STRUCTURE_MARKERS)
        constraint_score = _piece_constraint_score(piece, weights)
        score = 0.0
        if mode == "all":
            score = 1.0
        elif mode == "json-syntax":
            score = 100.0 if is_control else (4.0 if is_json_syntax else 0.0)
        elif mode == "constraint-paths":
            score = 100.0 if is_control else constraint_score
        else:
            score = max(100.0 if is_control else 0.0, 4.0 if is_json_syntax else 0.0, constraint_score)
        scores.append(score)
    return torch.tensor(scores, dtype=torch.float32, device=token_ids.device)


@dataclass(frozen=True)
class RealCacheStorageBudgetProfile:
    """Shape and format information needed for packed-cache byte budgeting."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int
    key_bits: int
    value_bits: int
    protected_bits: int
    residual_length: int
    group_size: int
    protection_target: str = "both"
    batch_size: int = 1

    def validate(self) -> None:
        if min(self.num_layers, self.num_kv_heads, self.head_dim, self.dtype_bytes, self.batch_size) <= 0:
            raise ValueError("cache storage dimensions must be positive")
        if self.residual_length <= 0 or self.group_size <= 0:
            raise ValueError("residual_length and group_size must be positive")
        if self.residual_length % self.group_size:
            raise ValueError("residual_length must be divisible by group_size")
        if self.head_dim % self.group_size:
            raise ValueError("head_dim must be divisible by group_size")
        if self.protection_target not in PROTECTION_TARGETS:
            raise ValueError(f"unknown protection target: {self.protection_target}")


def real_cache_storage_budget_profile(
    model,
    *,
    key_bits: int,
    value_bits: int,
    protected_bits: int | None,
    residual_length: int,
    group_size: int,
    protection_target: str,
) -> RealCacheStorageBudgetProfile:
    """Build an exact packed-storage budget profile from a causal LM."""
    config = model.config
    num_heads = int(getattr(config, "num_attention_heads", 0) or 0)
    hidden_size = int(getattr(config, "hidden_size", 0) or 0)
    parameter = next(model.parameters())
    profile = RealCacheStorageBudgetProfile(
        num_layers=int(getattr(config, "num_hidden_layers", 0) or 0),
        num_kv_heads=int(getattr(config, "num_key_value_heads", num_heads) or num_heads),
        head_dim=hidden_size // max(1, num_heads),
        dtype_bytes=parameter.element_size(),
        key_bits=int(key_bits),
        value_bits=int(value_bits),
        protected_bits=16 if protected_bits is None else int(protected_bits),
        residual_length=int(residual_length),
        group_size=int(group_size),
        protection_target=protection_target,
    )
    profile.validate()
    return profile


def _packed_storage_unit_bytes(
    profile: RealCacheStorageBudgetProfile,
    *,
    layout: str,
    bits: int,
) -> int:
    """Return bytes for one key token-group or one value token, all layers."""
    common = profile.num_layers * profile.batch_size * profile.num_kv_heads
    if layout == "keys":
        scalars = common * profile.group_size * profile.head_dim
        metadata_scalars = common * profile.head_dim * 2
    elif layout == "values":
        scalars = common * profile.head_dim
        metadata_scalars = common * (profile.head_dim // profile.group_size) * 2
    else:
        raise ValueError(f"unknown cache layout: {layout}")
    if bits >= 16:
        return scalars * profile.dtype_bytes
    return math.ceil(scalars * bits / 8) + (metadata_scalars * profile.dtype_bytes)


def real_cache_storage_bytes_for_mask(
    profile: RealCacheStorageBudgetProfile,
    seq_len: int,
    selected_mask: torch.Tensor | None = None,
) -> int:
    """Estimate persistent packed bytes using the same K/V layouts as the cache."""
    profile.validate()
    seq_len = max(0, int(seq_len))
    if seq_len == 0:
        return 0
    selected = (
        torch.zeros(seq_len, dtype=torch.bool)
        if selected_mask is None
        else selected_mask.detach().cpu().flatten().to(dtype=torch.bool)[:seq_len]
    )
    if selected.numel() < seq_len:
        selected = torch.cat([selected, torch.zeros(seq_len - selected.numel(), dtype=torch.bool)])

    key_prefix = seq_len - (seq_len % profile.residual_length)
    value_prefix = max(0, seq_len - profile.residual_length)
    key_groups = key_prefix // profile.group_size
    protected_key_groups = {
        index // profile.group_size
        for index in torch.nonzero(selected[:key_prefix], as_tuple=False).flatten().tolist()
    } if profile.protection_target in {"both", "keys"} else set()
    protected_value_tokens = int(selected[:value_prefix].sum().item()) if profile.protection_target in {"both", "values"} else 0

    key_base_cost = _packed_storage_unit_bytes(profile, layout="keys", bits=profile.key_bits)
    key_protected_cost = _packed_storage_unit_bytes(profile, layout="keys", bits=profile.protected_bits)
    value_base_cost = _packed_storage_unit_bytes(profile, layout="values", bits=profile.value_bits)
    value_protected_cost = _packed_storage_unit_bytes(profile, layout="values", bits=profile.protected_bits)
    key_bytes = ((key_groups - len(protected_key_groups)) * key_base_cost) + (
        len(protected_key_groups) * key_protected_cost
    )
    value_bytes = ((value_prefix - protected_value_tokens) * value_base_cost) + (
        protected_value_tokens * value_protected_cost
    )

    residual_tokens = (seq_len - key_prefix) + (seq_len - value_prefix)
    residual_bytes = (
        residual_tokens
        * profile.num_layers
        * profile.batch_size
        * profile.num_kv_heads
        * profile.head_dim
        * profile.dtype_bytes
    )
    return int(key_bytes + value_bytes + residual_bytes)


def real_cache_target_storage_bytes(
    profile: RealCacheStorageBudgetProfile,
    seq_len: int,
    target_average_bits: float,
) -> int:
    scalar_count = (
        2
        * max(0, int(seq_len))
        * profile.num_layers
        * profile.batch_size
        * profile.num_kv_heads
        * profile.head_dim
    )
    return math.floor((float(target_average_bits) * scalar_count) / 8)


def budgeted_protection_mask(
    candidate_mask: torch.Tensor | None,
    base_bits: float,
    protected_bits: int | None,
    residual_length: int,
    target_average_bits: float | None = None,
    order: str = "prefix",
    scores: torch.Tensor | None = None,
    storage_profile: RealCacheStorageBudgetProfile | None = None,
    random_seed: int = 0,
) -> torch.Tensor | None:
    """Select protected positions under an average-bit budget.

    The recent residual window is treated separately and remains FP16. This
    function allocates the extra budget only to older positions. With equal
    extra cost per protected token, the budgeted allocation is a thresholded
    selection over candidate positions.
    """
    if candidate_mask is None:
        return None
    if order not in PROTECTION_BUDGET_ORDERS:
        raise ValueError(f"unknown protection budget order: {order}")

    candidate_mask = candidate_mask.flatten().to(dtype=torch.bool)
    if target_average_bits is None or target_average_bits <= 0:
        return candidate_mask

    if storage_profile is not None:
        storage_profile.validate()
        seq_len = int(candidate_mask.numel())
        selected = torch.zeros_like(candidate_mask)
        baseline_bytes = real_cache_storage_bytes_for_mask(storage_profile, seq_len)
        target_bytes = real_cache_target_storage_bytes(storage_profile, seq_len, target_average_bits)
        if baseline_bytes >= target_bytes:
            return selected

        candidates = torch.nonzero(candidate_mask, as_tuple=False).flatten().tolist()
        if order == "score":
            if scores is None:
                raise ValueError("score-ordered protection requires scores")
            score_values = scores.to(device=candidate_mask.device, dtype=torch.float32).flatten()
            candidates.sort(key=lambda index: (-float(score_values[index].item()), int(index)))
        elif order == "recent":
            candidates.reverse()
        elif order == "random":
            random.Random(int(random_seed)).shuffle(candidates)

        current_bytes = baseline_bytes
        key_prefix = seq_len - (seq_len % storage_profile.residual_length)
        value_prefix = max(0, seq_len - storage_profile.residual_length)
        protected_key_groups: set[int] = set()
        protected_value_positions: set[int] = set()
        key_delta = _packed_storage_unit_bytes(
            storage_profile, layout="keys", bits=storage_profile.protected_bits
        ) - _packed_storage_unit_bytes(storage_profile, layout="keys", bits=storage_profile.key_bits)
        value_delta = _packed_storage_unit_bytes(
            storage_profile, layout="values", bits=storage_profile.protected_bits
        ) - _packed_storage_unit_bytes(storage_profile, layout="values", bits=storage_profile.value_bits)

        for index in candidates:
            marginal_bytes = 0
            key_group = index // storage_profile.group_size
            adds_key_group = (
                storage_profile.protection_target in {"both", "keys"}
                and index < key_prefix
                and key_group not in protected_key_groups
                and key_delta > 0
            )
            adds_value_token = (
                storage_profile.protection_target in {"both", "values"}
                and index < value_prefix
                and index not in protected_value_positions
                and value_delta > 0
            )
            if adds_key_group:
                marginal_bytes += key_delta
            if adds_value_token:
                marginal_bytes += value_delta
            if marginal_bytes <= 0 or current_bytes + marginal_bytes > target_bytes:
                continue
            selected[index] = True
            current_bytes += marginal_bytes
            if adds_key_group:
                protected_key_groups.add(key_group)
            if adds_value_token:
                protected_value_positions.add(index)
        return selected

    effective_protected_bits = 16 if protected_bits is None else protected_bits
    extra_bits_per_token = effective_protected_bits - base_bits
    if extra_bits_per_token <= 0:
        return candidate_mask

    seq_len = int(candidate_mask.numel())
    residual_count = min(max(0, int(residual_length)), seq_len)
    older_count = max(0, seq_len - residual_count)
    selected = torch.zeros_like(candidate_mask)
    if older_count <= 0:
        return selected

    baseline_bits = (older_count * base_bits) + (residual_count * 16)
    total_budget_bits = target_average_bits * seq_len
    extra_budget_bits = total_budget_bits - baseline_bits
    max_protected_older = math.floor(extra_budget_bits / extra_bits_per_token)
    if max_protected_older <= 0:
        return selected

    older_candidates = torch.nonzero(candidate_mask[:older_count], as_tuple=False).flatten()
    if older_candidates.numel() == 0:
        return selected
    if order == "score":
        if scores is None:
            raise ValueError("score-ordered protection requires scores")
        score_values = scores.to(device=candidate_mask.device, dtype=torch.float32).flatten()
        ranked_candidates = sorted(
            older_candidates.tolist(),
            key=lambda index: (-float(score_values[index].item()), int(index)),
        )
        older_candidates = torch.tensor(ranked_candidates, dtype=torch.long, device=candidate_mask.device)
    elif order == "recent":
        older_candidates = torch.flip(older_candidates, dims=[0])
    elif order == "random":
        shuffled = older_candidates.tolist()
        random.Random(int(random_seed)).shuffle(shuffled)
        older_candidates = torch.tensor(shuffled, dtype=torch.long, device=candidate_mask.device)
    chosen = older_candidates[:max_protected_older]
    selected[chosen] = True
    return selected


def quant_dim(axis: str) -> int:
    """Return the tensor dimension reduced to build a quantization scale.

    Cache tensors are [batch, kv_heads, tokens, head_dim].
    - per-token: one scale per token vector, reducing over head_dim.
    - per-channel: one scale per channel, reducing over token positions.
    """
    if axis == "per-token":
        return -1
    if axis == "per-channel":
        return -2
    raise ValueError(f"unknown quantization axis: {axis}")


def quantize_dynamic_cache_(
    past_key_values: Any,
    bits: int,
    key_bits: int | None = None,
    value_bits: int | None = None,
    key_axis: str = "per-channel",
    value_axis: str = "per-token",
    residual_length: int = 0,
    protected_positions: torch.Tensor | None = None,
    protected_bits: int | None = None,
    protection_target: str = "both",
    cache_quantization_mode: str = "repeated",
    quantized_prefix_lengths: list[int] | None = None,
    quantize_block_size: int = 1,
) -> Any:
    """In-place fake quantization for Hugging Face DynamicCache.

    Qwen2.5 uses grouped-query attention, so K/V have fewer heads than Q.
    Each layer stores:
      layer.keys:   [batch, kv_heads, tokens, head_dim]
      layer.values: [batch, kv_heads, tokens, head_dim]
    """
    effective_key_bits = bits if key_bits is None else key_bits
    effective_value_bits = bits if value_bits is None else value_bits
    if (effective_key_bits >= 16 and effective_value_bits >= 16) or past_key_values is None:
        return past_key_values
    if not hasattr(past_key_values, "layers"):
        raise TypeError(f"expected DynamicCache-like object, got {type(past_key_values)!r}")

    k_dim = quant_dim(key_axis)
    v_dim = quant_dim(value_axis)
    if cache_quantization_mode not in {"repeated", "blockwise"}:
        raise ValueError(f"unknown cache quantization mode: {cache_quantization_mode}")
    if protection_target not in PROTECTION_TARGETS:
        raise ValueError(f"unknown protection target: {protection_target}")

    if cache_quantization_mode == "blockwise" and quantized_prefix_lengths is None:
        raise ValueError("blockwise cache quantization requires quantized_prefix_lengths")

    for layer_index, layer in enumerate(past_key_values.layers):
        if not getattr(layer, "is_initialized", False):
            continue
        if cache_quantization_mode == "blockwise":
            while len(quantized_prefix_lengths) <= layer_index:
                quantized_prefix_lengths.append(0)
            previous_prefix = quantized_prefix_lengths[layer_index]
            new_prefixes = []
            if effective_key_bits < 16:
                layer.keys, key_new_prefix = fake_quantize_cache_tensor_blockwise(
                    layer.keys,
                    bits=effective_key_bits,
                    scale_dim=k_dim,
                    quantized_prefix_length=previous_prefix,
                    residual_length=residual_length,
                    block_size=quantize_block_size,
                    protected_positions=protected_positions if protection_target in {"both", "keys"} else None,
                    protected_bits=protected_bits,
                )
                new_prefixes.append(key_new_prefix)
            if effective_value_bits < 16:
                layer.values, value_new_prefix = fake_quantize_cache_tensor_blockwise(
                    layer.values,
                    bits=effective_value_bits,
                    scale_dim=v_dim,
                    quantized_prefix_length=previous_prefix,
                    residual_length=residual_length,
                    block_size=quantize_block_size,
                    protected_positions=protected_positions if protection_target in {"both", "values"} else None,
                    protected_bits=protected_bits,
                )
                new_prefixes.append(value_new_prefix)
            if new_prefixes:
                quantized_prefix_lengths[layer_index] = max(new_prefixes)
        else:
            if effective_key_bits < 16:
                layer.keys = fake_quantize_cache_tensor(
                    layer.keys,
                    bits=effective_key_bits,
                    scale_dim=k_dim,
                    residual_length=residual_length,
                    protected_positions=protected_positions if protection_target in {"both", "keys"} else None,
                    protected_bits=protected_bits,
                )
            if effective_value_bits < 16:
                layer.values = fake_quantize_cache_tensor(
                    layer.values,
                    bits=effective_value_bits,
                    scale_dim=v_dim,
                    residual_length=residual_length,
                    protected_positions=protected_positions if protection_target in {"both", "values"} else None,
                    protected_bits=protected_bits,
                )
    return past_key_values


def build_chat_prompt(tokenizer, user_prompt: str, system_prompt: str | None = None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


STRUCTEVAL_OFFICIAL_SUFFIX = (
    "\n\nIMPORTANT: Only output the required output format. You must start the "
    "format/code with <|BEGIN_CODE|> and end the format/code with  <|END_CODE|>. "
    "No other text output (explanation, comments, etc.) are allowed.  "
    "Do not use markdown code fences."
)


def build_structeval_official_prompt(query: str, model_name: str = "") -> str:
    """Apply the StructEval inference wrapper used by the official CLI."""
    qwen3_no_think = "\n\n/no_think" if model_name == "Qwen/Qwen3-4B" else ""
    return f"{query}{STRUCTEVAL_OFFICIAL_SUFFIX}{qwen3_no_think}"


def progress_iter(items: Iterable[T], total: int, enabled: bool = True, desc: str = "inference") -> Iterable[T]:
    """Wrap an iterable with tqdm when available, otherwise return it unchanged."""
    if not enabled:
        return items
    try:
        from tqdm.auto import tqdm

        return tqdm(items, total=total, desc=desc)
    except Exception:
        return items


def has_official_end_code_marker(text: str) -> bool:
    """Return whether generation has produced StructEval's requested end tag."""
    return "<|END_CODE|>" in text


def repeated_tail_ngram_count(token_ids: list[int], ngram_size: int) -> int:
    """Count how many times the final n-token pattern repeats at the sequence tail."""
    if ngram_size <= 0 or len(token_ids) < ngram_size * 2:
        return 1
    tail = token_ids[-ngram_size:]
    count = 1
    cursor = len(token_ids) - ngram_size * 2
    while cursor >= 0 and token_ids[cursor : cursor + ngram_size] == tail:
        count += 1
        cursor -= ngram_size
    return count


def resolve_generation_limit(model, input_length: int, requested_max_new_tokens: int) -> int:
    """Resolve the official unlimited setting against the model context window.

    StructEval specifies no explicit output-token cap. Autoregressive inference
    still has a finite context window, so a non-positive requested limit means
    "use every remaining model-context position" rather than an arbitrary
    benchmark-specific cap.
    """
    if requested_max_new_tokens > 0:
        return requested_max_new_tokens

    context_limit = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    if context_limit <= 0:
        raise ValueError("unlimited generation requires model.config.max_position_embeddings")
    remaining = context_limit - input_length
    if remaining <= 0:
        raise ValueError(
            f"prompt length {input_length} leaves no generation space in the {context_limit}-token context window"
        )
    return remaining


@torch.no_grad()
def generate_manual_kv(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    quantize_cache: bool = False,
    bits: int = 8,
    key_bits: int | None = None,
    value_bits: int | None = None,
    key_axis: str = "per-channel",
    value_axis: str = "per-token",
    residual_length: int = 0,
    structure_token_protection: str = "none",
    constraint_terms: list[str] | None = None,
    constraint_term_weights: dict[str, float] | None = None,
    protected_bits: int | None = None,
    target_average_bits: float | None = None,
    protection_budget_order: str = "prefix",
    protection_random_seed: int = 0,
    protection_target: str = "both",
    cache_quantization_mode: str = "repeated",
    quantize_block_size: int = 1,
    kv_group_size: int = 32,
    stop_on_end_code: bool = True,
    loop_ngram_size: int = 16,
    loop_repeat_threshold: int = 6,
    device: str | None = None,
) -> dict[str, Any]:
    """Manual autoregressive generation so we can modify past_key_values.

    Hugging Face `model.generate()` hides the cache update. For KV-cache research
    we use a manual loop:
      1. prefill the prompt and receive `past_key_values`
      2. optionally quantize the cache
      3. choose the next token from logits
      4. feed only that next token with the cache
      5. quantize the updated cache again
    """
    device = device or next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = encoded["input_ids"].shape[1]
    generation_limit = resolve_generation_limit(model, input_len, max_new_tokens)

    logits, past = None, None
    output_ids = encoded["input_ids"]
    next_input = encoded["input_ids"]
    protected_cache_positions = 0
    protected_cache_fraction = 0.0
    candidate_protected_cache_positions = 0
    candidate_protected_cache_fraction = 0.0
    protected_score_sum = 0.0
    candidate_score_sum = 0.0
    quantized_prefix_lengths: list[int] = []
    effective_key_bits = bits if key_bits is None else key_bits
    effective_value_bits = bits if value_bits is None else value_bits
    real_cache = None
    storage_profile = None
    protection_budget_estimated_storage_bytes = 0
    protection_budget_target_storage_bytes = 0
    stop_reason = "max_new_tokens"
    if quantize_cache and cache_quantization_mode == "real-blockwise":
        real_cache = RealQuantizedCache(
            RealQuantizedCacheConfig(
                bits=bits,
                key_bits=key_bits,
                value_bits=value_bits,
                residual_length=residual_length,
                group_size=kv_group_size,
                protected_bits=protected_bits,
                protection_target=protection_target,
            )
        )
        past = real_cache
        storage_profile = real_cache_storage_budget_profile(
            model,
            key_bits=effective_key_bits,
            value_bits=effective_value_bits,
            protected_bits=protected_bits,
            residual_length=residual_length,
            group_size=kv_group_size,
            protection_target=protection_target,
        )

    for step in range(generation_limit):
        protected_positions = None
        if quantize_cache:
            candidate_positions = structure_token_mask(
                tokenizer,
                output_ids[0],
                mode=structure_token_protection,
                constraint_terms=constraint_terms,
            )
            if candidate_positions is not None:
                candidate_protected_cache_positions = int(candidate_positions.sum().item())
                candidate_protected_cache_fraction = candidate_protected_cache_positions / max(
                    1,
                    int(candidate_positions.numel()),
                )
            candidate_scores = structure_token_scores(
                tokenizer,
                output_ids[0],
                mode=structure_token_protection,
                constraint_term_weights=constraint_term_weights,
            )
            if candidate_positions is not None:
                seq_len = int(candidate_positions.numel())
                older_count = max(0, seq_len - min(max(0, residual_length), seq_len))
                older_candidate_positions = candidate_positions.clone()
                if older_count < seq_len:
                    older_candidate_positions[older_count:] = False
                candidate_score_sum = float(candidate_scores[older_candidate_positions].sum().item())
            protected_positions = budgeted_protection_mask(
                candidate_positions,
                base_bits=(effective_key_bits + effective_value_bits) / 2,
                protected_bits=protected_bits,
                residual_length=residual_length,
                target_average_bits=target_average_bits,
                order=protection_budget_order,
                scores=candidate_scores,
                storage_profile=storage_profile if target_average_bits is not None else None,
                random_seed=protection_random_seed,
            )
            if protected_positions is not None:
                protected_cache_positions = int(protected_positions.sum().item())
                protected_cache_fraction = protected_cache_positions / max(1, int(protected_positions.numel()))
                eligible_protected_positions = protected_positions.clone()
                if older_count < eligible_protected_positions.numel():
                    eligible_protected_positions[older_count:] = False
                protected_score_sum = float(candidate_scores[eligible_protected_positions].sum().item())
            if real_cache is not None:
                real_cache.set_protected_positions(protected_positions, protected_bits=protected_bits)
            if storage_profile is not None and target_average_bits is not None:
                protection_budget_estimated_storage_bytes = real_cache_storage_bytes_for_mask(
                    storage_profile,
                    int(output_ids.shape[1]),
                    protected_positions,
                )
                protection_budget_target_storage_bytes = real_cache_target_storage_bytes(
                    storage_profile,
                    int(output_ids.shape[1]),
                    target_average_bits,
                )

        if step == 0 and real_cache is None:
            out = model(input_ids=next_input, use_cache=True, return_dict=True)
        else:
            out = model(input_ids=next_input, past_key_values=past, use_cache=True, return_dict=True)

        if real_cache is not None:
            # Cache.update() must expose the current full-precision residual to
            # attention. Only seal eligible K/V after every layer has consumed it.
            real_cache.finalize_after_attention()

        logits = out.logits[:, -1, :]
        past = out.past_key_values

        if quantize_cache and real_cache is None:
            quantize_dynamic_cache_(
                past,
                bits=bits,
                key_bits=key_bits,
                value_bits=value_bits,
                key_axis=key_axis,
                value_axis=value_axis,
                residual_length=residual_length,
                protected_positions=protected_positions,
                protected_bits=protected_bits,
                protection_target=protection_target,
                cache_quantization_mode=cache_quantization_mode,
                quantized_prefix_lengths=quantized_prefix_lengths,
                quantize_block_size=quantize_block_size,
            )

        if temperature and temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        output_ids = torch.cat([output_ids, next_token], dim=1)
        next_input = next_token

        if tokenizer.eos_token_id is not None and int(next_token.item()) == tokenizer.eos_token_id:
            stop_reason = "eos"
            break

        generated_tail = output_ids[0, input_len:].detach().cpu().tolist()
        if stop_on_end_code and step % 4 == 0:
            generated_text = tokenizer.decode(generated_tail, skip_special_tokens=True)
            if has_official_end_code_marker(generated_text):
                stop_reason = "official_end_code"
                break

        if loop_ngram_size > 0 and loop_repeat_threshold > 1:
            repeats = repeated_tail_ngram_count(generated_tail, loop_ngram_size)
            if repeats >= loop_repeat_threshold:
                stop_reason = f"repeated_{loop_ngram_size}gram_x{repeats}"
                break

    else:
        stop_reason = "max_new_tokens"

    if stop_reason == "max_new_tokens" and output_ids.shape[1] - input_len < generation_limit:
        stop_reason = "unknown"

    generated_ids = output_ids[0, input_len:]
    if hasattr(past, "get_seq_length"):
        cache_sequence_length = int(past.get_seq_length())
    else:
        cache_sequence_length = max(0, int(output_ids.shape[1]) - 1)
    real_cache_storage_bytes = real_cache.storage_bytes() if real_cache is not None else 0
    model_config = model.config
    attention_heads = int(getattr(model_config, "num_attention_heads", 0) or 0)
    cache_num_layers = int(getattr(model_config, "num_hidden_layers", 0) or 0)
    cache_kv_heads = int(getattr(model_config, "num_key_value_heads", attention_heads) or attention_heads)
    cache_head_dim = int(getattr(model_config, "hidden_size", 0) or 0) // max(1, attention_heads)
    cache_dtype_bytes = next(model.parameters()).element_size()
    cache_scalar_count = 2 * cache_sequence_length * cache_num_layers * cache_kv_heads * cache_head_dim
    fp16_cache_bytes = cache_scalar_count * 2
    persistent_cache_storage_bytes = (
        real_cache_storage_bytes if real_cache is not None else cache_scalar_count * cache_dtype_bytes
    )
    return {
        "text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "token_ids": generated_ids.detach().cpu().tolist(),
        "input_tokens": input_len,
        "generated_tokens": len(generated_ids),
        "generation_limit": generation_limit,
        "stop_reason": stop_reason,
        "key_bits": effective_key_bits,
        "value_bits": effective_value_bits,
        "protected_cache_positions": protected_cache_positions,
        "protected_cache_fraction": protected_cache_fraction,
        "candidate_protected_cache_positions": candidate_protected_cache_positions,
        "candidate_protected_cache_fraction": candidate_protected_cache_fraction,
        "protected_score_sum": protected_score_sum,
        "candidate_score_sum": candidate_score_sum,
        "protected_score_fraction": protected_score_sum / candidate_score_sum if candidate_score_sum > 0 else 0.0,
        "protected_bits": protected_bits,
        "target_average_bits": target_average_bits,
        "protection_budget_model": "group-aware-packed-bytes-v1" if storage_profile is not None else "token-count-v1",
        "protection_budget_estimated_storage_bytes": protection_budget_estimated_storage_bytes,
        "protection_budget_target_storage_bytes": protection_budget_target_storage_bytes,
        "protection_budget_order": protection_budget_order,
        "protection_random_seed": protection_random_seed,
        "protection_target": protection_target,
        "constraint_terms": constraint_terms or [],
        "constraint_term_weights": constraint_term_weights or {},
        "cache_quantization_mode": cache_quantization_mode,
        "quantize_block_size": quantize_block_size,
        "kv_group_size": kv_group_size if real_cache is not None else None,
        "cache_algorithm_revision": (
            real_quantized_cache.CACHE_ALGORITHM_REVISION if real_cache is not None else "not-applicable"
        ),
        "prefill_cache_policy": "full_precision_attention_then_store" if real_cache is not None else "mode_dependent",
        "mean_quantized_prefix_length": (
            real_cache.mean_quantized_prefix_length()
            if real_cache is not None
            else (sum(quantized_prefix_lengths) / len(quantized_prefix_lengths) if quantized_prefix_lengths else 0.0)
        ),
        "cache_sequence_length": cache_sequence_length,
        "cache_dtype_bytes": cache_dtype_bytes,
        "persistent_cache_storage_bytes": persistent_cache_storage_bytes,
        "persistent_cache_compression_ratio_vs_fp16": (
            fp16_cache_bytes / persistent_cache_storage_bytes if persistent_cache_storage_bytes > 0 else 0.0
        ),
        "persistent_cache_effective_bits_per_scalar": (
            (persistent_cache_storage_bytes * 8) / cache_scalar_count if cache_scalar_count > 0 else 0.0
        ),
        "cache_storage_semantics": (
            "packed-affine-persistent-cache"
            if real_cache is not None
            else ("full-width-fake-quantized-tensors" if quantize_cache else "full-precision-dynamic-cache")
        ),
        "real_cache_storage_bytes": real_cache_storage_bytes,
        "real_cache_storage_breakdown": real_cache.storage_breakdown() if real_cache is not None else {},
        "real_cache_mean_quantized_lengths": real_cache.mean_quantized_prefix_lengths() if real_cache is not None else {},
        "fp16_equivalent_cache_bytes": fp16_cache_bytes,
        "real_cache_compression_ratio_vs_fp16": (
            fp16_cache_bytes / real_cache_storage_bytes if real_cache_storage_bytes > 0 else 1.0
        ),
        "real_cache_effective_bits_per_scalar": (
            (real_cache_storage_bytes * 8) / cache_scalar_count
            if real_cache_storage_bytes > 0 and cache_scalar_count > 0
            else float(cache_dtype_bytes * 8)
        ),
    }


STRUCTEVAL_SAMPLING_MODES = ("head", "stratified")


def _structeval_stratum(row: dict[str, Any], output_type: str | None) -> str:
    input_name = str(row.get("input_type") or "unknown")
    if output_type is not None:
        return input_name
    output_name = str(row.get("output_type") or "unknown")
    return f"{input_name}->{output_name}"


def load_structeval_rows(
    path: str | Path,
    limit: int = 10,
    output_type: str | None = "JSON",
    sampling: str = "stratified",
    seed: int = 42,
) -> list[dict]:
    """Load a reproducible StructEval subset.

    StructEval stores tasks in contiguous task-family blocks. Prefix sampling
    therefore selects only one or two families. Stratified sampling shuffles
    within each input/output family and draws from every family round-robin.
    """
    if sampling not in STRUCTEVAL_SAMPLING_MODES:
        raise ValueError(f"unknown StructEval sampling mode: {sampling}")
    if limit <= 0:
        return []

    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if output_type is not None and row.get("output_type") != output_type:
                continue
            rows.append(row)

    if sampling == "head":
        selected = rows[:limit]
    else:
        rng = random.Random(seed)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(_structeval_stratum(row, output_type), []).append(row)
        for group_rows in grouped.values():
            rng.shuffle(group_rows)

        strata = sorted(grouped)
        rng.shuffle(strata)
        offsets = {stratum: 0 for stratum in strata}
        selected = []
        while len(selected) < min(limit, len(rows)):
            added = False
            for stratum in strata:
                offset = offsets[stratum]
                group_rows = grouped[stratum]
                if offset >= len(group_rows):
                    continue
                selected.append(group_rows[offset])
                offsets[stratum] = offset + 1
                added = True
                if len(selected) >= limit:
                    break
            if not added:
                break
        rng.shuffle(selected)

    result: list[dict[str, Any]] = []
    for selection_index, source_row in enumerate(selected):
        row = dict(source_row)
        row["_structeval_sampling"] = sampling
        row["_structeval_seed"] = seed
        row["_structeval_stratum"] = _structeval_stratum(row, output_type)
        row["_structeval_selection_index"] = selection_index
        result.append(row)
    return result


def structeval_selection_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("_structeval_stratum") or "unknown") for row in rows).items()))


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_structeval_manifest_rows(source_path: str | Path, manifest_path: str | Path) -> list[dict[str, Any]]:
    """Load exactly the task IDs frozen in a checked StructEval manifest."""
    source_path = Path(source_path)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("source_sha256")
    actual_hash = _file_sha256(source_path)
    if expected_hash and expected_hash != actual_hash:
        raise ValueError(
            f"StructEval source hash does not match manifest: expected {expected_hash}, got {actual_hash}"
        )

    all_rows: dict[str, dict[str, Any]] = {}
    with source_path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            all_rows[str(row.get("task_id"))] = row

    result: list[dict[str, Any]] = []
    for task in manifest.get("tasks", []):
        task_id = str(task.get("task_id"))
        if task_id not in all_rows:
            raise ValueError(f"manifest task {task_id} is not present in {source_path}")
        row = dict(all_rows[task_id])
        for field in ("input_type", "output_type"):
            if task.get(field) is not None and row.get(field) != task[field]:
                raise ValueError(f"manifest task {task_id} has mismatched {field}")
        row["_structeval_sampling"] = f"manifest:{manifest.get('sampling', 'unknown')}"
        row["_structeval_seed"] = manifest.get("seed")
        row["_structeval_stratum"] = task.get("stratum") or _structeval_stratum(row, row.get("output_type"))
        row["_structeval_selection_index"] = task.get("selection_index")
        row["_structeval_manifest"] = manifest_path.name
        row["_structeval_manifest_source_sha256"] = actual_hash
        result.append(row)
    return result


def required_token_match_rate(output: str, required_items: list[Any]) -> float:
    if not required_items:
        return 0.0
    required = [str(item) for item in required_items]
    return sum(item in output for item in required) / len(required)


def extract_json_candidate(text: str) -> str:
    """Return the most likely JSON object/array from a model response."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return text.strip()

    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        return text[start:].strip()
    return text[start : end + 1].strip()


def normalize_structeval_generation(text: str) -> str:
    """Match StructEval's light normalization before extracting generated code."""
    try:
        text = text.replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("<think>\n\n</think>\n\n", "")
        text = codecs.decode(text, "unicode_escape")
    except Exception:
        pass
    return text


def extract_structeval_official_code(text: str, output_type: str = "json") -> str:
    """Extract generated code with the same priority as StructEval render_utils.

    Priority:
      1. <|BEGIN_CODE|> ... <|END_CODE|> (closing tag optional)
      2. fenced code block, matching any fence header
      3. raw text fallback
    """
    text = normalize_structeval_generation(text)
    output_type = output_type.lower()

    begin_end_pat = (
        r"<\|BEGIN_CODE\|\>[ \t]*\n?"
        r"(?P<payload1>.*?)"
        r"(?:<\|END_CODE\|\>|$)"
    )
    fence_pat = (
        rf"```(?:{re.escape(output_type)}|[^\n]*)[ \t]*\n"
        r"(?P<payload2>.*?)"
        r"(?:```|$)"
    )

    pattern = rf"(?:{begin_end_pat})|(?:{fence_pat})"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        code = text.strip()
    else:
        code = (match.group("payload1") or match.group("payload2")).strip()

    # StructEval's non-renderable extractor performs this second fence check.
    fence_match = re.search(fence_pat, text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        code = fence_match.group("payload2").strip()
    return code


def parse_json_output(text: str) -> tuple[Any | None, str]:
    candidate = extract_json_candidate(text)
    try:
        return json.loads(candidate), candidate
    except json.JSONDecodeError:
        return None, candidate


def parse_json_output_with_error(
    text: str,
    official_structeval: bool = False,
    output_type: str = "json",
) -> tuple[Any | None, str, str]:
    candidate = (
        extract_structeval_official_code(text, output_type=output_type)
        if official_structeval
        else extract_json_candidate(text)
    )
    try:
        return json.loads(candidate), candidate, ""
    except json.JSONDecodeError as exc:
        return None, candidate, str(exc)


def has_unclosed_code_fence(text: str) -> bool:
    return text.count("```") % 2 == 1


def path_parts(path: str) -> list[str | int]:
    """Convert a StructEval path such as a.b[0].c or a.*.c into parts."""
    parts: list[str | int] = []
    for segment in path.split("."):
        for match in re.finditer(r"([^\[\]]+)|\[(\d+)\]", segment):
            key, index = match.groups()
            if key is not None:
                parts.append(key)
            else:
                parts.append(int(index))
    return parts


def has_path(value: Any, parts: list[str | int]) -> bool:
    if not parts:
        return True

    head, *tail = parts
    if head == "*":
        if isinstance(value, list):
            return any(has_path(item, tail) for item in value)
        return False

    if isinstance(head, int):
        return isinstance(value, list) and 0 <= head < len(value) and has_path(value[head], tail)

    return isinstance(value, dict) and head in value and has_path(value[head], tail)


def required_json_path_rate(parsed: Any, required_items: list[Any]) -> float:
    if parsed is None or not required_items:
        return 0.0
    return sum(has_path(parsed, path_parts(str(item))) for item in required_items) / len(required_items)


def required_json_path_details(parsed: Any, required_items: list[Any]) -> dict[str, list[str] | float]:
    required = [str(item) for item in required_items]
    present = [item for item in required if parsed is not None and has_path(parsed, path_parts(item))]
    missing = [item for item in required if item not in present]
    rate = len(present) / len(required) if required else 0.0
    return {
        "present_required_paths": present,
        "missing_required_paths": missing,
        "json_required_path_rate": rate,
    }


def structeval_render_score(parsed: Any) -> float:
    """Official non-renderable render score: valid, non-empty parsed output."""
    return 1.0 if parsed is not None and bool(parsed) else 0.0


def structeval_nonrenderable_score(render_score: float, key_validation_score: float) -> float:
    """Official final score for non-renderable tasks."""
    return round((0.2 * render_score) + (0.8 * key_validation_score), 2)


def structeval_checkpoint_key(
    row: dict[str, Any],
    bits: int,
    key_bits: int | None,
    value_bits: int | None,
) -> str:
    task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.get("task_id") or "missing"))
    selection_index = int(row.get("_structeval_selection_index") or 0)
    effective_key_bits = bits if key_bits is None else key_bits
    effective_value_bits = bits if value_bits is None else value_bits
    return f"{selection_index:04d}_{task_id}_K{effective_key_bits}_V{effective_value_bits}"


def run_structeval_smoke(
    model,
    tokenizer,
    device: str,
    rows: list[dict],
    bits_list: list[int],
    max_new_tokens: int,
    key_bits: int | None = None,
    value_bits: int | None = None,
    official_prompt: bool = False,
    official_extraction: bool = False,
    model_name: str = "",
    show_progress: bool = True,
    residual_length: int = 0,
    structure_token_protection: str = "none",
    protection_signal_source: str = "prompt-visible",
    protected_bits: int | None = None,
    target_average_bits: float | None = None,
    protection_budget_order: str = "prefix",
    protection_random_seed: int = 0,
    protection_target: str = "both",
    cache_quantization_mode: str = "repeated",
    quantize_block_size: int = 1,
    kv_group_size: int = 32,
    stop_on_end_code: bool = True,
    loop_ngram_size: int = 16,
    loop_repeat_threshold: int = 6,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    run_metadata: dict[str, Any] | None = None,
    generation_seed: int = 123,
) -> list[dict]:
    results: list[dict[str, Any]] = []
    jobs = [(row, bits) for row in rows for bits in bits_list]
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir else None
    checkpoint_rows: dict[str, dict[str, Any]] = {}
    invocation_started = time.perf_counter()
    if checkpoint_root is not None:
        if run_metadata is None or not run_metadata.get("run_fingerprint"):
            raise ValueError("checkpointed runs require run metadata with a run fingerprint")
        metadata_path = checkpoint_root / "run_metadata.json"
        task_dir = checkpoint_root / "tasks"
        task_dir.mkdir(parents=True, exist_ok=True)
        if metadata_path.exists():
            existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing_metadata.get("run_fingerprint") != run_metadata.get("run_fingerprint"):
                raise ValueError("checkpoint run fingerprint does not match the requested experiment")
            if not resume and any(task_dir.glob("*.json")):
                raise FileExistsError(
                    f"checkpoint rows already exist in {task_dir}; pass resume=True to continue this run"
                )
            run_metadata = existing_metadata
        else:
            atomic_write_json(metadata_path, run_metadata)
        if resume:
            for checkpoint_path in sorted(task_dir.glob("*.json")):
                checkpoint_row = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                key = str(checkpoint_row.get("checkpoint_key") or checkpoint_path.stem)
                if checkpoint_row.get("run_fingerprint") != run_metadata.get("run_fingerprint"):
                    raise ValueError(f"checkpoint row {checkpoint_path} has a mismatched run fingerprint")
                checkpoint_rows[key] = checkpoint_row
        atomic_write_json(
            checkpoint_root / "status.json",
            {
                "status": "running",
                "run_id": run_metadata.get("run_id"),
                "total_jobs": len(jobs),
                "completed_jobs": len(checkpoint_rows),
                "resumed": bool(checkpoint_rows),
                "updated_at_utc": utc_now_iso(),
            },
        )
    iterator = progress_iter(jobs, total=len(jobs), enabled=show_progress, desc="StructEval inference")
    for row, bits in iterator:
        checkpoint_key = structeval_checkpoint_key(row, bits, key_bits, value_bits)
        if checkpoint_key in checkpoint_rows:
            results.append(checkpoint_rows[checkpoint_key])
            continue
        prompt = str(row["query"])
        if official_prompt:
            prompt = build_structeval_official_prompt(prompt, model_name=model_name)
        required = row.get("raw_output_metric") or []
        constraint_terms, constraint_term_weights = resolve_protection_signals(
            str(row["query"]),
            required,
            protection_signal_source,
        )
        if hasattr(iterator, "set_postfix_str"):
            label = f"K{key_bits if key_bits is not None else bits}/V{value_bits if value_bits is not None else bits}"
            iterator.set_postfix_str(f"task={row.get('task_id')} bits={label}", refresh=False)
        effective_key_bits = bits if key_bits is None else key_bits
        effective_value_bits = bits if value_bits is None else value_bits
        quantize = effective_key_bits < 16 or effective_value_bits < 16
        task_started_at = utc_now_iso()
        task_started = time.perf_counter()
        torch.manual_seed(generation_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(generation_seed)
        try:
            out = generate_manual_kv(
                model,
                tokenizer,
                build_chat_prompt(tokenizer, prompt),
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                quantize_cache=quantize,
                bits=bits,
                key_bits=key_bits,
                value_bits=value_bits,
                residual_length=residual_length,
                structure_token_protection=structure_token_protection,
                constraint_terms=constraint_terms,
                constraint_term_weights=constraint_term_weights,
                protected_bits=protected_bits,
                target_average_bits=target_average_bits,
                protection_budget_order=protection_budget_order,
                protection_random_seed=protection_random_seed,
                protection_target=protection_target,
                cache_quantization_mode=cache_quantization_mode,
                quantize_block_size=quantize_block_size,
                kv_group_size=kv_group_size,
                stop_on_end_code=stop_on_end_code,
                loop_ngram_size=loop_ngram_size,
                loop_repeat_threshold=loop_repeat_threshold,
                device=device,
            )
        except BaseException as error:
            if checkpoint_root is not None:
                atomic_write_json(
                    checkpoint_root / "status.json",
                    {
                        "status": "failed",
                        "run_id": run_metadata.get("run_id") if run_metadata else None,
                        "current_checkpoint_key": checkpoint_key,
                        "completed_jobs": len(results),
                        "total_jobs": len(jobs),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "updated_at_utc": utc_now_iso(),
                    },
                )
            raise
        task_duration_seconds = time.perf_counter() - task_started
        if row.get("output_type") == "JSON":
            parsed_json, json_candidate, json_parse_error = parse_json_output_with_error(
                out["text"],
                official_structeval=official_extraction,
                output_type="json",
            )
        else:
            parsed_json, json_candidate, json_parse_error = None, "", ""
        path_details = required_json_path_details(parsed_json, required) if row.get("output_type") == "JSON" else {}
        official_render_score = structeval_render_score(parsed_json) if row.get("output_type") == "JSON" else 0.0
        official_key_validation_score = float(path_details.get("json_required_path_rate") or 0.0)
        metadata = run_metadata or {}
        environment_metadata = metadata.get("environment") or {}
        model_metadata = metadata.get("model") or {}
        result_row = {
                "checkpoint_key": checkpoint_key,
                "run_id": metadata.get("run_id"),
                "run_fingerprint": metadata.get("run_fingerprint"),
                "combined_source_sha256": metadata.get("combined_source_sha256"),
                "model_config_sha256": model_metadata.get("config_sha256"),
                "device_name": environment_metadata.get("device_name"),
                "torch_version": environment_metadata.get("torch"),
                "transformers_version": environment_metadata.get("transformers"),
                "generation_seed": generation_seed,
                "generation_temperature": 0.0,
                "task_started_at_utc": task_started_at,
                "task_duration_seconds": round(task_duration_seconds, 6),
                "task_id": row.get("task_id"),
                "query": row.get("query", ""),
                "feature_requirements": row.get("feature_requirements", ""),
                "task_name": row.get("task_name"),
                "input_type": row.get("input_type"),
                "output_type": row.get("output_type"),
                "query_example": row.get("query_example", ""),
                "VQA": row.get("VQA") or [],
                "raw_output_metric": row.get("raw_output_metric") or [],
                "rendering": bool(row.get("rendering", False)),
                "structeval_sampling": row.get("_structeval_sampling", "provided"),
                "structeval_seed": row.get("_structeval_seed"),
                "structeval_stratum": row.get("_structeval_stratum"),
                "structeval_selection_index": row.get("_structeval_selection_index"),
                "structeval_manifest": row.get("_structeval_manifest"),
                "structeval_manifest_source_sha256": row.get("_structeval_manifest_source_sha256"),
                "expected_required_paths": [str(item) for item in required],
                "expected_vqa": row.get("VQA") or [],
                "bits": bits,
                "key_bits": out["key_bits"],
                "value_bits": out["value_bits"],
                "input_tokens": out["input_tokens"],
                "num_hidden_layers": int(getattr(model.config, "num_hidden_layers", 0) or 0),
                "num_key_value_heads": int(
                    getattr(
                        model.config,
                        "num_key_value_heads",
                        getattr(model.config, "num_attention_heads", 0),
                    )
                    or 0
                ),
                "head_dim": int(
                    (getattr(model.config, "hidden_size", 0) or 0)
                    // max(1, int(getattr(model.config, "num_attention_heads", 1) or 1))
                ),
                "quantize_cache": quantize,
                "residual_length": residual_length,
                "structure_token_protection": structure_token_protection,
                "protection_signal_source": protection_signal_source,
                "protection_method_label": (
                    "uniform-kivi4"
                    if structure_token_protection == "none"
                    else (
                        "random-protection"
                        if structure_token_protection == "all" and out["protection_budget_order"] == "random"
                        else (
                            "recency-protection"
                            if structure_token_protection == "all" and out["protection_budget_order"] == "recent"
                            else (
                                "oracle-path-upper-bound"
                                if protection_signal_source == "oracle-required-paths"
                                else "structure-aware-protection"
                            )
                        )
                    )
                ),
                "uses_evaluator_path_leakage": protection_signal_source == "oracle-required-paths",
                "protected_bits": out["protected_bits"],
                "target_average_bits": out["target_average_bits"],
                "protection_budget_model": out["protection_budget_model"],
                "protection_budget_estimated_storage_bytes": out[
                    "protection_budget_estimated_storage_bytes"
                ],
                "protection_budget_target_storage_bytes": out["protection_budget_target_storage_bytes"],
                "protection_budget_order": out["protection_budget_order"],
                "protection_random_seed": out.get("protection_random_seed", protection_random_seed),
                "protection_target": out["protection_target"],
                "constraint_terms": out["constraint_terms"],
                "constraint_term_weights": out["constraint_term_weights"],
                "cache_quantization_mode": out["cache_quantization_mode"],
                "quantize_block_size": out["quantize_block_size"],
                "kv_group_size": out["kv_group_size"],
                "cache_algorithm_revision": out["cache_algorithm_revision"],
                "prefill_cache_policy": out["prefill_cache_policy"],
                "cache_sequence_length": out["cache_sequence_length"],
                "cache_dtype_bytes": out["cache_dtype_bytes"],
                "cache_storage_semantics": out["cache_storage_semantics"],
                "persistent_cache_storage_bytes": out["persistent_cache_storage_bytes"],
                "persistent_cache_compression_ratio_vs_fp16": out[
                    "persistent_cache_compression_ratio_vs_fp16"
                ],
                "persistent_cache_effective_bits_per_scalar": out[
                    "persistent_cache_effective_bits_per_scalar"
                ],
                "mean_quantized_prefix_length": out["mean_quantized_prefix_length"],
                "real_cache_storage_bytes": out["real_cache_storage_bytes"],
                "real_cache_storage_breakdown": out["real_cache_storage_breakdown"],
                "real_cache_mean_quantized_lengths": out["real_cache_mean_quantized_lengths"],
                "fp16_equivalent_cache_bytes": out["fp16_equivalent_cache_bytes"],
                "real_cache_compression_ratio_vs_fp16": out["real_cache_compression_ratio_vs_fp16"],
                "real_cache_effective_bits_per_scalar": out["real_cache_effective_bits_per_scalar"],
                "stop_reason": out["stop_reason"],
                "stop_on_end_code": stop_on_end_code,
                "loop_ngram_size": loop_ngram_size,
                "loop_repeat_threshold": loop_repeat_threshold,
                "protected_cache_positions": out["protected_cache_positions"],
                "protected_cache_fraction": out["protected_cache_fraction"],
                "candidate_protected_cache_positions": out["candidate_protected_cache_positions"],
                "candidate_protected_cache_fraction": out["candidate_protected_cache_fraction"],
                "protected_score_sum": out["protected_score_sum"],
                "candidate_score_sum": out["candidate_score_sum"],
                "protected_score_fraction": out["protected_score_fraction"],
                "official_prompt": official_prompt,
                "official_extraction": official_extraction,
                "json_parse_success": parsed_json is not None if row.get("output_type") == "JSON" else None,
                "json_required_path_rate": path_details.get("json_required_path_rate", None)
                if row.get("output_type") == "JSON"
                else None,
                "structeval_render_score": official_render_score,
                "structeval_key_validation_score": official_key_validation_score,
                "structeval_final_eval_score": structeval_nonrenderable_score(
                    official_render_score,
                    official_key_validation_score,
                )
                if row.get("output_type") == "JSON"
                else 0.0,
                "present_required_paths": path_details.get("present_required_paths", []),
                "missing_required_paths": path_details.get("missing_required_paths", []),
                "required_match_rate": required_token_match_rate(out["text"], required),
                "generated_tokens": out["generated_tokens"],
                "generation_limit": out["generation_limit"],
                "reached_max_new_tokens": out["stop_reason"] == "max_new_tokens",
                "has_unclosed_code_fence": has_unclosed_code_fence(out["text"]),
                "json_parse_error": json_parse_error,
                "output_text": out["text"],
                "generation": out["text"],
                "json_candidate": json_candidate,
                "json_candidate_preview": json_candidate[:500],
                "output_preview": out["text"][:500],
            }
        results.append(result_row)
        if checkpoint_root is not None:
            atomic_write_json(checkpoint_root / "tasks" / f"{checkpoint_key}.json", result_row)
            atomic_write_json(
                checkpoint_root / "status.json",
                {
                    "status": "running",
                    "run_id": metadata.get("run_id"),
                    "last_completed_checkpoint_key": checkpoint_key,
                    "completed_jobs": len(results),
                    "total_jobs": len(jobs),
                    "updated_at_utc": utc_now_iso(),
                },
            )
    if checkpoint_root is not None:
        completed_at = utc_now_iso()
        run_metadata = dict(run_metadata or {})
        run_metadata.update(
            {
                "run_completed_at_utc": completed_at,
                "completed_results": len(results),
                "wall_time_seconds_last_invocation": round(time.perf_counter() - invocation_started, 6),
            }
        )
        atomic_write_json(checkpoint_root / "run_metadata.json", run_metadata)
        atomic_write_json(
            checkpoint_root / "status.json",
            {
                "status": "complete",
                "run_id": run_metadata.get("run_id"),
                "completed_jobs": len(results),
                "total_jobs": len(jobs),
                "updated_at_utc": completed_at,
            },
        )
    return results


def summarize_results(results: list[dict]) -> dict[str, dict[str, float]]:
    by_setting: dict[str, list[dict]] = {}
    for row in results:
        bits = int(row["bits"])
        key_bits = int(row.get("key_bits") if row.get("key_bits") is not None else bits)
        value_bits = int(row.get("value_bits") if row.get("value_bits") is not None else bits)
        residual_length = int(row.get("residual_length") or 0)
        protection = str(row.get("structure_token_protection") or "none")
        protected_bits = row.get("protected_bits")
        target_average_bits = row.get("target_average_bits")
        protection_target = str(row.get("protection_target") or "both")
        mode = str(row.get("cache_quantization_mode") or "repeated")
        block_size = int(row.get("quantize_block_size") or 1)
        key = (
            f"bits={bits},K={key_bits},V={value_bits},mode={mode},"
            f"residual={residual_length},block={block_size},protection={protection},"
            f"target={protection_target},"
            f"protected_bits={protected_bits if protected_bits is not None else 'fp16'},"
            f"target_avg={target_average_bits if target_average_bits is not None else 'none'}"
        )
        by_setting.setdefault(key, []).append(row)

    summary = {}
    for setting, rows in sorted(by_setting.items()):
        json_rows = [row for row in rows if row.get("json_parse_success") is not None]
        summary[setting] = {
            "n": float(len(rows)),
            "json_parse_success_rate": sum(bool(row["json_parse_success"]) for row in json_rows) / len(json_rows)
            if json_rows
            else 0.0,
            "mean_json_required_path_rate": sum(float(row["json_required_path_rate"] or 0.0) for row in json_rows)
            / len(json_rows)
            if json_rows
            else 0.0,
            "mean_structeval_final_eval_score": sum(float(row.get("structeval_final_eval_score") or 0.0) for row in json_rows)
            / len(json_rows)
            if json_rows
            else 0.0,
            "max_token_hit_rate": sum(bool(row.get("reached_max_new_tokens")) for row in rows) / len(rows)
            if rows
            else 0.0,
            "mean_required_text_match_rate": sum(float(row["required_match_rate"]) for row in rows) / len(rows)
            if rows
            else 0.0,
            "mean_protected_cache_fraction": sum(float(row.get("protected_cache_fraction") or 0.0) for row in rows)
            / len(rows)
            if rows
            else 0.0,
            "mean_candidate_protected_cache_fraction": sum(
                float(row.get("candidate_protected_cache_fraction") or 0.0) for row in rows
            )
            / len(rows)
            if rows
            else 0.0,
            "mean_protected_score_fraction": sum(float(row.get("protected_score_fraction") or 0.0) for row in rows)
            / len(rows)
            if rows
            else 0.0,
            "mean_quantized_prefix_length": sum(float(row.get("mean_quantized_prefix_length") or 0.0) for row in rows)
            / len(rows)
            if rows
            else 0.0,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--prompt", default="Return a JSON object with keys name and age for Alice, age 7. Only JSON.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="Maximum generated tokens; use 0 for StructEval's context-limited unlimited setting.",
    )
    parser.add_argument("--bits", type=int, nargs="+", default=[16, 8, 4, 2])
    parser.add_argument("--key-bits", type=int, default=None)
    parser.add_argument("--value-bits", type=int, default=None)
    parser.add_argument("--structeval-jsonl", default="")
    parser.add_argument(
        "--structeval-manifest",
        default="",
        help="Optional checked task manifest. When set, it overrides limit/sampling selection.",
    )
    parser.add_argument("--structeval-limit", type=int, default=3)
    parser.add_argument("--structeval-sampling", choices=STRUCTEVAL_SAMPLING_MODES, default="stratified")
    parser.add_argument("--structeval-seed", type=int, default=42)
    parser.add_argument(
        "--output-type",
        default="JSON",
        help="Filter to one StructEval output type, or use ALL for the complete benchmark.",
    )
    parser.add_argument("--out", default="toy_kv_experiments/results/qwen_structeval_smoke.json")
    parser.add_argument(
        "--checkpoint-dir",
        default="",
        help="Atomic per-task checkpoint directory; defaults to <out>.checkpoints.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume matching task checkpoints.")
    parser.add_argument("--generation-seed", type=int, default=123)
    parser.add_argument("--official-structeval-prompt", action="store_true")
    parser.add_argument("--official-structeval-extraction", action="store_true")
    parser.add_argument("--official-model-name", default="")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--residual-length", type=int, default=0)
    parser.add_argument("--structure-token-protection", choices=STRUCTURE_TOKEN_PROTECTION_MODES, default="none")
    parser.add_argument(
        "--protection-signal-source",
        choices=PROTECTION_SIGNAL_SOURCES,
        default="prompt-visible",
        help="Use prompt-visible identifiers by default; oracle-required-paths is an explicitly labeled upper bound.",
    )
    parser.add_argument(
        "--allow-oracle-protection",
        action="store_true",
        help="Explicitly permit evaluator-required paths as a labeled oracle upper bound.",
    )
    parser.add_argument("--protected-bits", type=int, default=None)
    parser.add_argument("--target-average-bits", type=float, default=None)
    parser.add_argument("--protection-budget-order", choices=PROTECTION_BUDGET_ORDERS, default="prefix")
    parser.add_argument(
        "--protection-random-seed",
        type=int,
        default=0,
        help="Deterministic seed for random protection-order controls.",
    )
    parser.add_argument("--protection-target", choices=PROTECTION_TARGETS, default="both")
    parser.add_argument("--disable-end-code-stop", action="store_true")
    parser.add_argument("--loop-ngram-size", type=int, default=16)
    parser.add_argument("--loop-repeat-threshold", type=int, default=6)
    parser.add_argument(
        "--cache-quantization-mode",
        choices=["repeated", "blockwise", "real-blockwise"],
        default="repeated",
    )
    parser.add_argument("--quantize-block-size", type=int, default=1)
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=32,
        help="KIVI group size for real-blockwise cache storage; keys group tokens and values group channels.",
    )
    args = parser.parse_args()
    if args.protection_signal_source == "oracle-required-paths" and not args.allow_oracle_protection:
        parser.error("oracle-required-paths requires --allow-oracle-protection to prevent evaluator leakage")

    model, tokenizer, device = load_qwen(args.model_dir)
    print("device:", device)
    print("model:", args.model_dir)

    if args.structeval_jsonl:
        output_type_filter = None if args.output_type.strip().upper() == "ALL" else args.output_type
        if args.structeval_manifest:
            rows = load_structeval_manifest_rows(args.structeval_jsonl, args.structeval_manifest)
            print("StructEval manifest:", args.structeval_manifest)
        else:
            rows = load_structeval_rows(
                args.structeval_jsonl,
                limit=args.structeval_limit,
                output_type=output_type_filter,
                sampling=args.structeval_sampling,
                seed=args.structeval_seed,
            )
            print("StructEval sampling:", args.structeval_sampling)
            print("StructEval seed:", args.structeval_seed)
        print("StructEval strata:", json.dumps(structeval_selection_counts(rows), sort_keys=True))
        out_path = Path(args.out)
        checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path(f"{out_path}.checkpoints")
        protocol = {
            "benchmark_scope": (
                "StructEval full benchmark"
                if output_type_filter is None
                else f"StructEval {output_type_filter} output subset"
            ),
            "task_ids": [str(row.get("task_id")) for row in rows],
            "task_manifest": Path(args.structeval_manifest).name if args.structeval_manifest else None,
            "task_manifest_source_sha256": rows[0].get("_structeval_manifest_source_sha256") if rows else None,
            "official_prompt_wrapper": bool(args.official_structeval_prompt),
            "official_extraction": bool(args.official_structeval_extraction),
            "temperature": 0.0,
            "max_new_tokens": None if args.max_new_tokens <= 0 else args.max_new_tokens,
            "generation_limit_policy": (
                "remaining-model-context" if args.max_new_tokens <= 0 else "fixed-cap"
            ),
            "stop_on_end_code": not args.disable_end_code_stop,
            "loop_ngram_size": args.loop_ngram_size,
            "loop_repeat_threshold": args.loop_repeat_threshold,
            "bits": args.bits,
            "key_bits": args.key_bits,
            "value_bits": args.value_bits,
            "residual_length": args.residual_length,
            "kv_group_size": args.kv_group_size,
            "cache_quantization_mode": args.cache_quantization_mode,
            "cache_algorithm_revision": (
                real_quantized_cache.CACHE_ALGORITHM_REVISION
                if args.cache_quantization_mode == "real-blockwise"
                else "not-applicable"
            ),
            "structure_token_protection": args.structure_token_protection,
            "protection_signal_source": args.protection_signal_source,
            "uses_evaluator_path_leakage": args.protection_signal_source == "oracle-required-paths",
            "protected_bits": args.protected_bits,
            "target_average_bits": args.target_average_bits,
            "protection_budget_order": args.protection_budget_order,
            "protection_random_seed": args.protection_random_seed,
            "protection_target": args.protection_target,
        }
        run_metadata = build_run_metadata(
            model_dir=resolve_model_dir(args.model_dir),
            model_name=args.official_model_name,
            device=device,
            protocol=protocol,
            generation_seed=args.generation_seed,
        )
        print("run id:", run_metadata["run_id"])
        print("run fingerprint:", run_metadata["run_fingerprint"])
        print("checkpoint dir:", checkpoint_dir)
        results = run_structeval_smoke(
            model,
            tokenizer,
            device,
            rows=rows,
            bits_list=args.bits,
            max_new_tokens=args.max_new_tokens,
            key_bits=args.key_bits,
            value_bits=args.value_bits,
            official_prompt=args.official_structeval_prompt,
            official_extraction=args.official_structeval_extraction,
            model_name=args.official_model_name,
            show_progress=not args.no_progress,
            residual_length=args.residual_length,
            structure_token_protection=args.structure_token_protection,
            protection_signal_source=args.protection_signal_source,
            protected_bits=args.protected_bits,
            target_average_bits=args.target_average_bits,
            protection_budget_order=args.protection_budget_order,
            protection_random_seed=args.protection_random_seed,
            protection_target=args.protection_target,
            cache_quantization_mode=args.cache_quantization_mode,
            quantize_block_size=args.quantize_block_size,
            kv_group_size=args.kv_group_size,
            stop_on_end_code=not args.disable_end_code_stop,
            loop_ngram_size=args.loop_ngram_size,
            loop_repeat_threshold=args.loop_repeat_threshold,
            checkpoint_dir=checkpoint_dir,
            resume=args.resume,
            run_metadata=run_metadata,
            generation_seed=args.generation_seed,
        )
        atomic_write_json(out_path, results)
        print(f"saved {len(results)} results to {out_path}")
        print("summary:")
        print(json.dumps(summarize_results(results), ensure_ascii=False, indent=2))
        for row in results[: min(8, len(results))]:
            print(json.dumps(row, ensure_ascii=False, indent=2)[:1000])
        return

    chat_prompt = build_chat_prompt(tokenizer, args.prompt, system_prompt="You are a precise assistant.")
    for bits in args.bits:
        effective_key_bits = bits if args.key_bits is None else args.key_bits
        effective_value_bits = bits if args.value_bits is None else args.value_bits
        quantize = effective_key_bits < 16 or effective_value_bits < 16
        out = generate_manual_kv(
            model,
            tokenizer,
            chat_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            quantize_cache=quantize,
            bits=bits,
            key_bits=args.key_bits,
            value_bits=args.value_bits,
            residual_length=args.residual_length,
            structure_token_protection=args.structure_token_protection,
            protected_bits=args.protected_bits,
            target_average_bits=args.target_average_bits,
            protection_budget_order=args.protection_budget_order,
            protection_random_seed=args.protection_random_seed,
            protection_target=args.protection_target,
            cache_quantization_mode=args.cache_quantization_mode,
            quantize_block_size=args.quantize_block_size,
            kv_group_size=args.kv_group_size,
            stop_on_end_code=not args.disable_end_code_stop,
            loop_ngram_size=args.loop_ngram_size,
            loop_repeat_threshold=args.loop_repeat_threshold,
            device=device,
        )
        if not quantize:
            label = "full KV"
        elif args.cache_quantization_mode == "real-blockwise":
            label = f"real packed KV K{effective_key_bits}/V{effective_value_bits}"
        else:
            label = f"fake KV K{effective_key_bits}/V{effective_value_bits}"
        print("\n" + "=" * 80)
        print(label)
        print("=" * 80)
        print(out["text"])


if __name__ == "__main__":
    main()
