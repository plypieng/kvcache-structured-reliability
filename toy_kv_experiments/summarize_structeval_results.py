from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONFIG_FIELDS = (
    "bits",
    "key_bits",
    "value_bits",
    "cache_quantization_mode",
    "cache_algorithm_revision",
    "residual_length",
    "kv_group_size",
    "quantize_block_size",
    "structure_token_protection",
    "protection_signal_source",
    "protection_target",
    "protected_bits",
    "target_average_bits",
    "protection_budget_model",
    "protection_budget_order",
    "official_prompt",
    "official_extraction",
    "stop_on_end_code",
    "loop_ngram_size",
    "loop_repeat_threshold",
    "output_type",
    "structeval_manifest",
    "structeval_manifest_source_sha256",
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        rows = data.get("rows") or data.get("results")
        if isinstance(rows, list):
            return rows
    raise ValueError(f"unsupported result JSON shape: {path}")


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def row_estimated_fp16_cache_bytes(row: dict[str, Any]) -> float:
    """Estimate final FP16 cache tensor bytes for this generated sequence."""
    recorded = float(row.get("fp16_equivalent_cache_bytes") or 0.0)
    if recorded > 0:
        return recorded
    input_tokens = float(row.get("input_tokens") or 0.0)
    generated_tokens = float(row.get("generated_tokens") or 0.0)
    total_tokens = input_tokens + generated_tokens
    if total_tokens <= generated_tokens:
        quantized_prefix = float(row.get("mean_quantized_prefix_length") or 0.0)
        residual_length = float(row.get("residual_length") or 0.0)
        total_tokens = max(total_tokens, quantized_prefix + residual_length)
    layers = float(row.get("num_hidden_layers") or 28)
    kv_heads = float(row.get("num_key_value_heads") or 4)
    head_dim = float(row.get("head_dim") or 128)
    return total_tokens * layers * kv_heads * head_dim * 2.0 * 2.0


def _value(row: dict[str, Any], field: str) -> str:
    defaults = {
        "key_bits": row.get("bits"),
        "value_bits": row.get("bits"),
        "cache_quantization_mode": "repeated",
        "cache_algorithm_revision": (
            "legacy-materialize-before-seal-v1"
            if row.get("quantize_cache") and row.get("cache_quantization_mode") == "real-blockwise"
            else "not-applicable"
        ),
        "structure_token_protection": "none",
        "protection_signal_source": "legacy-implicit-oracle",
        "protection_target": "both",
        "protected_bits": "fp16",
        "target_average_bits": "none",
        "protection_budget_model": "legacy-token-count",
        "protection_budget_order": "prefix",
        "official_prompt": False,
        "official_extraction": False,
        "stop_on_end_code": True,
        "loop_ngram_size": 16,
        "loop_repeat_threshold": 6,
        "output_type": "JSON",
        "structeval_manifest": "legacy-prefix",
        "structeval_manifest_source_sha256": "unknown",
        "kv_group_size": "none",
        "quantize_block_size": 1,
    }
    value = row.get(field, defaults.get(field))
    return json.dumps(value, sort_keys=True, default=str)


def row_configuration_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(_value(row, field) for field in CONFIG_FIELDS)


def task_set_sha256(rows: list[dict[str, Any]]) -> str:
    task_ids = sorted(str(row.get("task_id")) for row in rows)
    return hashlib.sha256("\n".join(task_ids).encode("utf-8")).hexdigest()[:12]


def mean_breakdown_bytes(rows: list[dict[str, Any]]) -> dict[str, float]:
    names = ("payload", "metadata", "residual")
    totals = {name: [] for name in names}
    for row in rows:
        breakdown = row.get("real_cache_storage_breakdown") or {}
        payload = sum(float(value) for key, value in breakdown.items() if key.endswith("payload_bytes"))
        metadata = sum(
            float(value)
            for key, value in breakdown.items()
            if key.endswith("scale_bytes") or key.endswith("minimum_bytes")
        )
        residual = sum(float(value) for key, value in breakdown.items() if key.endswith("residual_bytes"))
        totals["payload"].append(payload)
        totals["metadata"].append(metadata)
        totals["residual"].append(residual)
    return {name: mean(values) for name, values in totals.items()}


def summarize_rows(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty row group")
    n = len(rows)
    persistent_cache_storage = [
        float(row.get("persistent_cache_storage_bytes") or row.get("real_cache_storage_bytes") or 0.0)
        for row in rows
    ]
    fp16_cache_storage = [row_estimated_fp16_cache_bytes(row) for row in rows]
    mean_persistent_cache_bytes = mean(persistent_cache_storage)
    mean_fp16_cache_bytes = mean(fp16_cache_storage)
    cache_compression = mean_fp16_cache_bytes / mean_persistent_cache_bytes if mean_persistent_cache_bytes > 0 else 1.0
    breakdown = mean_breakdown_bytes(rows)
    configuration = dict(zip(CONFIG_FIELDS, row_configuration_key(rows[0]), strict=True))
    return {
        "file": path.name,
        "n": n,
        "task_set_sha256": task_set_sha256(rows),
        "input_type_counts": ",".join(
            f"{name}:{count}" for name, count in sorted(Counter(str(row.get("input_type") or "unknown") for row in rows).items())
        ),
        "structeval_sampling": sorted({str(row.get("structeval_sampling") or "legacy-prefix") for row in rows}),
        "structeval_seed": sorted({str(row.get("structeval_seed")) for row in rows}),
        "bits": configuration["bits"],
        "key_bits": configuration["key_bits"],
        "value_bits": configuration["value_bits"],
        "mode": configuration["cache_quantization_mode"],
        "cache_algorithm_revision": configuration["cache_algorithm_revision"],
        "residual_length": configuration["residual_length"],
        "kv_group_size": configuration["kv_group_size"],
        "structure_token_protection": configuration["structure_token_protection"],
        "protection_signal_source": configuration["protection_signal_source"],
        "protection_target": configuration["protection_target"],
        "protected_bits": configuration["protected_bits"],
        "target_average_bits": configuration["target_average_bits"],
        "parse_success": f"{sum(bool(row.get('json_parse_success')) for row in rows)}/{n}",
        "strict_full_paths": f"{sum((row.get('json_required_path_rate') or 0) == 1 for row in rows)}/{n}",
        "max_token_hits": f"{sum(bool(row.get('reached_max_new_tokens')) for row in rows)}/{n}",
        "mean_path_rate": round(mean([float(row.get("json_required_path_rate") or 0.0) for row in rows]), 4),
        "mean_score": round(mean([float(row.get("structeval_final_eval_score") or 0.0) for row in rows]), 4),
        "mean_tokens": round(mean([float(row.get("generated_tokens") or 0.0) for row in rows]), 1),
        "mean_quantized_prefix_length": round(mean([float(row.get("mean_quantized_prefix_length") or 0.0) for row in rows]), 1),
        "mean_persistent_cache_storage_mb": round(mean_persistent_cache_bytes / (1024 * 1024), 3),
        "mean_real_cache_storage_mb": round(
            mean([float(row.get("real_cache_storage_bytes") or 0.0) for row in rows]) / (1024 * 1024),
            3,
        ),
        "mean_fp16_cache_estimate_mb": round(mean_fp16_cache_bytes / (1024 * 1024), 3),
        "mean_cache_compression_vs_fp16": round(cache_compression, 3),
        "mean_effective_storage_bits_per_scalar": round(
            mean(
                [
                    float(
                        row.get("persistent_cache_effective_bits_per_scalar")
                        or row.get("real_cache_effective_bits_per_scalar")
                        or 0.0
                    )
                    for row in rows
                ]
            ),
            3,
        ),
        "mean_payload_mb": round(breakdown["payload"] / (1024 * 1024), 3),
        "mean_metadata_mb": round(breakdown["metadata"] / (1024 * 1024), 3),
        "mean_residual_mb": round(breakdown["residual"] / (1024 * 1024), 3),
        "mean_protected_score_fraction": round(mean([float(row.get("protected_score_fraction") or 0.0) for row in rows]), 4),
        "_configuration_key": row_configuration_key(rows[0]),
    }


def summarize_file(path: Path) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in load_rows(path):
        groups[row_configuration_key(row)].append(row)
    return [summarize_rows(path, rows) for _, rows in sorted(groups.items())]


def summary_config_key(summary: dict[str, Any]) -> tuple[str, ...]:
    return tuple(summary["_configuration_key"]) + (str(summary["task_set_sha256"]),)


def latest_by_config(paths: list[Path]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, ...], tuple[float, dict[str, Any]]] = {}
    for path in paths:
        if not path.exists():
            continue
        for summary in summarize_file(path):
            key = summary_config_key(summary)
            mtime = path.stat().st_mtime
            if key not in latest or mtime > latest[key][0]:
                latest[key] = (mtime, summary)
    return [item[1] for item in sorted(latest.values(), key=lambda item: (item[1]["file"], item[1]["bits"]))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize StructEval result rows by compatible treatment.")
    parser.add_argument("paths", nargs="+", help="Result JSON files or glob patterns.")
    parser.add_argument(
        "--latest-by-config",
        action="store_true",
        help="Keep the newest result only when treatment configuration and task set are identical.",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.paths:
        matches = sorted(Path().glob(pattern)) if any(char in pattern for char in "*?[]") else [Path(pattern)]
        paths.extend(matches)
    summaries = latest_by_config(paths) if args.latest_by_config else [
        summary for path in paths if path.exists() for summary in summarize_file(path)
    ]
    if not summaries:
        raise SystemExit("no result files found")

    columns = [
        "file",
        "n",
        "task_set_sha256",
        "input_type_counts",
        "structeval_sampling",
        "structeval_seed",
        "key_bits",
        "value_bits",
        "mode",
        "cache_algorithm_revision",
        "residual_length",
        "kv_group_size",
        "structure_token_protection",
        "protection_signal_source",
        "parse_success",
        "strict_full_paths",
        "max_token_hits",
        "mean_path_rate",
        "mean_score",
        "mean_persistent_cache_storage_mb",
        "mean_payload_mb",
        "mean_metadata_mb",
        "mean_residual_mb",
        "mean_effective_storage_bits_per_scalar",
        "mean_cache_compression_vs_fp16",
    ]
    widths = {column: max(len(column), *(len(str(row[column])) for row in summaries)) for column in columns}
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in summaries:
        print("  ".join(str(row[column]).ljust(widths[column]) for column in columns))


if __name__ == "__main__":
    main()
