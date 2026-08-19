from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from toy_kv_experiments.summarize_structeval_official import (
    CATEGORY_ORDER,
    EXPECTED_CATEGORY_COUNTS,
    load_rows,
    structeval_official_category,
)


EXPECTED_ROWS = sum(EXPECTED_CATEGORY_COUNTS.values())
REQUIRED_INFERENCE_FIELDS = (
    "task_id",
    "input_type",
    "output_type",
    "query",
    "generation",
    "rendering",
)


def validate_inference(
    rows: list[dict[str, Any]],
    *,
    expected_key_bits: int | None = None,
    expected_value_bits: int | None = None,
    expected_task_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Validate that an inference artifact covers the official benchmark once."""
    task_ids = [str(row.get("task_id", "")) for row in rows]
    id_counts = Counter(task_ids)
    duplicate_task_ids = sorted(task_id for task_id, count in id_counts.items() if count > 1)
    missing_task_id_rows = [index for index, task_id in enumerate(task_ids) if not task_id]

    categories = Counter(structeval_official_category(row) for row in rows)
    category_counts = {
        category: {
            "count": categories.get(category, 0),
            "expected_count": EXPECTED_CATEGORY_COUNTS[category],
        }
        for category in CATEGORY_ORDER
    }
    unexpected_categories = sorted(set(categories) - set(CATEGORY_ORDER))

    missing_fields: dict[str, list[int]] = {}
    for field in REQUIRED_INFERENCE_FIELDS:
        missing = [index for index, row in enumerate(rows) if field not in row]
        if missing:
            missing_fields[field] = missing

    bit_mismatches: dict[str, list[str]] = {}
    if expected_key_bits is not None:
        mismatches = [
            task_id
            for task_id, row in zip(task_ids, rows, strict=True)
            if row.get("key_bits") != expected_key_bits
        ]
        if mismatches:
            bit_mismatches["key_bits"] = mismatches
    if expected_value_bits is not None:
        mismatches = [
            task_id
            for task_id, row in zip(task_ids, rows, strict=True)
            if row.get("value_bits") != expected_value_bits
        ]
        if mismatches:
            bit_mismatches["value_bits"] = mismatches

    category_scope_complete = all(
        values["count"] == values["expected_count"] for values in category_counts.values()
    )
    exact_task_order = expected_task_ids is None or task_ids == expected_task_ids
    missing_reference_task_ids = (
        [] if expected_task_ids is None else sorted(set(expected_task_ids) - set(task_ids))
    )
    unexpected_task_ids = (
        [] if expected_task_ids is None else sorted(set(task_ids) - set(expected_task_ids))
    )
    valid = (
        len(rows) == EXPECTED_ROWS
        and len(id_counts) == EXPECTED_ROWS
        and not duplicate_task_ids
        and not missing_task_id_rows
        and category_scope_complete
        and not unexpected_categories
        and not missing_fields
        and not bit_mismatches
        and exact_task_order
    )

    return {
        "artifact_type": "inference",
        "valid": valid,
        "dataset_rows": len(rows),
        "expected_rows": EXPECTED_ROWS,
        "unique_task_ids": len(id_counts),
        "duplicate_task_ids": duplicate_task_ids,
        "missing_task_id_rows": missing_task_id_rows,
        "categories": category_counts,
        "unexpected_categories": unexpected_categories,
        "missing_fields": missing_fields,
        "bit_mismatches": bit_mismatches,
        "reference_task_count": len(expected_task_ids) if expected_task_ids is not None else None,
        "exact_reference_task_order": exact_task_order,
        "missing_reference_task_ids": missing_reference_task_ids,
        "unexpected_task_ids": unexpected_task_ids,
    }


def validate_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Validate a leaderboard summary before treating a condition as scored."""
    categories = summary.get("categories", {})
    category_scope_complete = all(
        categories.get(category, {}).get("count") == expected
        for category, expected in EXPECTED_CATEGORY_COUNTS.items()
    )
    valid = (
        summary.get("official_scope_complete") is True
        and summary.get("dataset_rows") == EXPECTED_ROWS
        and summary.get("scored_rows") == EXPECTED_ROWS
        and summary.get("unscored_rows") == 0
        and category_scope_complete
    )
    return {
        "artifact_type": "leaderboard-summary",
        "valid": valid,
        "dataset_rows": summary.get("dataset_rows"),
        "scored_rows": summary.get("scored_rows"),
        "unscored_rows": summary.get("unscored_rows"),
        "categories": categories,
        "official_scope_complete": summary.get("official_scope_complete"),
    }


def load_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("StructEval summary must be a JSON object")
    return payload


def load_dataset_task_ids(path: str | Path) -> list[str]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not row.get("task_id"):
            raise ValueError(f"invalid StructEval dataset row at line {line_number}")
        rows.append(str(row["task_id"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail unless a StructEval inference or score artifact covers the official set."
    )
    subparsers = parser.add_subparsers(dest="artifact_type", required=True)

    inference_parser = subparsers.add_parser("inference")
    inference_parser.add_argument("path")
    inference_parser.add_argument("--expected-key-bits", type=int)
    inference_parser.add_argument("--expected-value-bits", type=int)
    inference_parser.add_argument(
        "--dataset",
        help="Official StructEval JSONL whose exact task ID order must match.",
    )

    summary_parser = subparsers.add_parser("summary")
    summary_parser.add_argument("path")

    args = parser.parse_args()
    if args.artifact_type == "inference":
        result = validate_inference(
            load_rows(args.path),
            expected_key_bits=args.expected_key_bits,
            expected_value_bits=args.expected_value_bits,
            expected_task_ids=load_dataset_task_ids(args.dataset) if args.dataset else None,
        )
    else:
        result = validate_summary(load_object(args.path))

    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
