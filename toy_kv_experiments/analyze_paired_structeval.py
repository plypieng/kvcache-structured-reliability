"""Compare a compressed StructEval run against its paired FP16 baseline."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected a list of result rows in {path}")
    return data


def index_rows(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    indexed = {str(row.get("task_id")): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"{name} contains duplicate or missing task IDs")
    return indexed


def assert_paired(fp16_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    fp16 = index_rows(fp16_rows, "FP16")
    candidate = index_rows(candidate_rows, "candidate")
    if set(fp16) != set(candidate):
        missing_candidate = sorted(set(fp16) - set(candidate))[:5]
        missing_fp16 = sorted(set(candidate) - set(fp16))[:5]
        raise ValueError(
            "paired files have different task IDs; "
            f"missing from candidate={missing_candidate}, missing from FP16={missing_fp16}"
        )
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for task_id in sorted(fp16):
        base, compressed = fp16[task_id], candidate[task_id]
        for field in ("input_type", "output_type", "structeval_manifest_source_sha256"):
            base_value = base.get(field)
            candidate_value = compressed.get(field)
            if base_value and candidate_value and base_value != candidate_value:
                raise ValueError(f"task {task_id} differs in {field}: {base_value!r} != {candidate_value!r}")
        pairs.append((base, compressed))
    return pairs


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def parsed_json_candidate(row: dict[str, Any]) -> Any | None:
    if not row.get("json_parse_success"):
        return None
    candidate = row.get("json_candidate")
    if not isinstance(candidate, str):
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def flatten_json_leaves(value: Any, path: str = "$") -> dict[str, str]:
    """Map JSON leaf paths to canonical values for baseline-relative comparison."""
    if isinstance(value, dict):
        if not value:
            return {path: "{}"}
        flattened: dict[str, str] = {}
        for key in sorted(value):
            flattened.update(flatten_json_leaves(value[key], f"{path}.{key}"))
        return flattened
    if isinstance(value, list):
        if not value:
            return {path: "[]"}
        flattened = {}
        for index, item in enumerate(value):
            flattened.update(flatten_json_leaves(item, f"{path}[{index}]"))
        return flattened
    return {path: json.dumps(value, sort_keys=True, ensure_ascii=False)}


def fp16_reference_semantic_agreement(
    fp16_row: dict[str, Any],
    candidate_row: dict[str, Any],
) -> dict[str, Any] | None:
    """Compare JSON leaf path-values to FP16; this is not a ground-truth metric."""
    fp16_json = parsed_json_candidate(fp16_row)
    if fp16_json is None:
        return None
    candidate_json = parsed_json_candidate(candidate_row)
    reference = flatten_json_leaves(fp16_json)
    candidate = flatten_json_leaves(candidate_json) if candidate_json is not None else {}
    reference_pairs = set(reference.items())
    candidate_pairs = set(candidate.items())
    pair_overlap = len(reference_pairs & candidate_pairs)
    precision = pair_overlap / len(candidate_pairs) if candidate_pairs else 0.0
    recall = pair_overlap / len(reference_pairs) if reference_pairs else 0.0
    leaf_pair_f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    common_paths = set(reference) & set(candidate)
    common_path_value_accuracy = (
        sum(reference[path] == candidate[path] for path in common_paths) / len(common_paths)
        if common_paths
        else 0.0
    )
    path_union = set(reference) | set(candidate)
    return {
        "fp16_reference_leaf_pair_f1": leaf_pair_f1,
        "fp16_reference_leaf_pair_precision": precision,
        "fp16_reference_leaf_pair_recall": recall,
        "fp16_reference_common_path_value_accuracy": common_path_value_accuracy,
        "fp16_reference_path_jaccard": len(common_paths) / len(path_union) if path_union else 1.0,
        "fp16_reference_exact_json_match": candidate_json == fp16_json,
        "fp16_reference_leaf_count": len(reference),
        "candidate_leaf_count": len(candidate),
    }


def exact_mcnemar_pvalue(base: list[bool], candidate: list[bool]) -> float:
    """Two-sided exact binomial McNemar p-value for paired binary outcomes."""
    b = sum(a and not c for a, c in zip(base, candidate, strict=True))
    c = sum(not a and d for a, d in zip(base, candidate, strict=True))
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(b, c) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci(
    deltas: list[float],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not deltas:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(deltas)
    estimates = sorted(mean([deltas[rng.randrange(n)] for _ in range(n)]) for _ in range(samples))
    low = estimates[int(0.025 * (samples - 1))]
    high = estimates[int(0.975 * (samples - 1))]
    return low, high


def bool_metric(row: dict[str, Any], name: str) -> bool:
    if name == "parse":
        return bool(row.get("json_parse_success"))
    if name == "full_paths":
        return float(row.get("json_required_path_rate") or 0.0) == 1.0
    raise ValueError(name)


def pair_summary(pairs: list[tuple[dict[str, Any], dict[str, Any]]], bootstrap_samples: int, seed: int) -> dict[str, Any]:
    base_parse = [bool_metric(base, "parse") for base, _ in pairs]
    candidate_parse = [bool_metric(candidate, "parse") for _, candidate in pairs]
    base_paths = [float(base.get("json_required_path_rate") or 0.0) for base, _ in pairs]
    candidate_paths = [float(candidate.get("json_required_path_rate") or 0.0) for _, candidate in pairs]
    base_scores = [float(base.get("structeval_final_eval_score") or 0.0) for base, _ in pairs]
    candidate_scores = [float(candidate.get("structeval_final_eval_score") or 0.0) for _, candidate in pairs]
    fp16_success_indices = [index for index, success in enumerate(base_parse) if success]
    induced_parse_failures = [
        pairs[index]
        for index in fp16_success_indices
        if not candidate_parse[index]
    ]
    deltas = [candidate - base for base, candidate in zip(base_paths, candidate_paths, strict=True)]
    score_deltas = [candidate - base for base, candidate in zip(base_scores, candidate_scores, strict=True)]
    semantic = [
        agreement
        for base, candidate in pairs
        if (agreement := fp16_reference_semantic_agreement(base, candidate)) is not None
    ]
    return {
        "n": len(pairs),
        "fp16_parse_success": sum(base_parse),
        "candidate_parse_success": sum(candidate_parse),
        "compression_induced_parse_failures": len(induced_parse_failures),
        "compression_induced_parse_failure_rate_given_fp16_success": (
            len(induced_parse_failures) / len(fp16_success_indices) if fp16_success_indices else 0.0
        ),
        "fp16_full_path_success": sum(bool_metric(base, "full_paths") for base, _ in pairs),
        "candidate_full_path_success": sum(bool_metric(candidate, "full_paths") for _, candidate in pairs),
        "mean_path_rate_fp16": mean(base_paths),
        "mean_path_rate_candidate": mean(candidate_paths),
        "mean_path_rate_delta_candidate_minus_fp16": mean(deltas),
        "mean_path_rate_delta_95ci": paired_bootstrap_ci(deltas, bootstrap_samples, seed),
        "mean_structeval_score_fp16": mean(base_scores),
        "mean_structeval_score_candidate": mean(candidate_scores),
        "mean_structeval_score_delta_candidate_minus_fp16": mean(score_deltas),
        "mean_structeval_score_delta_95ci": paired_bootstrap_ci(score_deltas, bootstrap_samples, seed + 1),
        "parse_mcnemar_pvalue": exact_mcnemar_pvalue(base_parse, candidate_parse),
        "fp16_reference_semantic_n": len(semantic),
        "mean_fp16_reference_leaf_pair_f1": mean(
            [float(item["fp16_reference_leaf_pair_f1"]) for item in semantic]
        ),
        "mean_fp16_reference_common_path_value_accuracy": mean(
            [float(item["fp16_reference_common_path_value_accuracy"]) for item in semantic]
        ),
        "fp16_reference_exact_json_matches": sum(
            bool(item["fp16_reference_exact_json_match"]) for item in semantic
        ),
    }


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": row.get("task_id"),
        "task_name": row.get("task_name"),
        "input_type": row.get("input_type"),
        "parse": row.get("json_parse_success"),
        "path_rate": row.get("json_required_path_rate"),
        "score": row.get("structeval_final_eval_score"),
        "stop_reason": row.get("stop_reason"),
        "output_preview": row.get("output_preview"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp16", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = assert_paired(load_rows(args.fp16), load_rows(args.candidate))
    overall = pair_summary(pairs, args.bootstrap_samples, args.seed)
    by_family: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in pairs:
        by_family[str(pair[0].get("input_type") or "unknown")].append(pair)
    induced_examples = [
        {"fp16": compact_row(base), "candidate": compact_row(candidate)}
        for base, candidate in pairs
        if bool_metric(base, "parse") and not bool_metric(candidate, "parse")
    ]
    semantic_examples = []
    for base, candidate in pairs:
        agreement = fp16_reference_semantic_agreement(base, candidate)
        if agreement is None or agreement["fp16_reference_exact_json_match"]:
            continue
        semantic_examples.append(
            {
                "task_id": base.get("task_id"),
                "task_name": base.get("task_name"),
                "input_type": base.get("input_type"),
                **agreement,
                "fp16_output_preview": base.get("output_preview"),
                "candidate_output_preview": candidate.get("output_preview"),
            }
        )
    semantic_examples.sort(key=lambda item: float(item["fp16_reference_leaf_pair_f1"]))
    report = {
        "fp16_file": args.fp16.name,
        "candidate_file": args.candidate.name,
        "task_manifest": pairs[0][0].get("structeval_manifest") if pairs else None,
        "task_manifest_source_sha256": pairs[0][0].get("structeval_manifest_source_sha256") if pairs else None,
        "overall": overall,
        "by_input_type": {
            family: pair_summary(family_pairs, args.bootstrap_samples, args.seed)
            for family, family_pairs in sorted(by_family.items())
        },
        "compression_induced_parse_failure_examples": induced_examples[:10],
        "fp16_reference_semantic_disagreement_examples": semantic_examples[:10],
        "semantic_metric_note": (
            "Leaf path-value agreement uses paired FP16 output as a behavioral reference; "
            "it is not ground truth and is not part of the official StructEval score."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
