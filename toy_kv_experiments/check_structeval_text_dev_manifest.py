from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from toy_kv_experiments.pretrained_kv_quantization import load_structeval_manifest_rows
from toy_kv_experiments.summarize_structeval_official import structeval_official_category


TEXT_CATEGORIES = ("T-generation", "T-conversion")


def validate_rows(rows: list[dict[str, Any]], *, expected_rows: int = 20) -> dict[str, Any]:
    if expected_rows <= 0 or expected_rows % len(TEXT_CATEGORIES):
        raise ValueError("expected_rows must be a positive even number")

    expected_per_category = expected_rows // len(TEXT_CATEGORIES)
    task_ids = [str(row.get("task_id") or "") for row in rows]
    categories = Counter(structeval_official_category(row) for row in rows)
    category_counts = {category: categories.get(category, 0) for category in TEXT_CATEGORIES}
    task_types = Counter(f"{row.get('input_type')}->{row.get('output_type')}" for row in rows)
    normalized_outputs = {str(row.get("output_type") or "").lower() for row in rows}
    rendering_rows = [str(row.get("task_id")) for row in rows if bool(row.get("rendering", False))]
    unexpected_categories = {
        category: count for category, count in categories.items() if category not in TEXT_CATEGORIES
    }

    valid = (
        len(rows) == expected_rows
        and len(set(task_ids)) == expected_rows
        and all(task_ids)
        and all(count == expected_per_category for count in category_counts.values())
        and not unexpected_categories
        and not rendering_rows
    )
    return {
        "valid": valid,
        "rows": len(rows),
        "unique_task_ids": len(set(task_ids)),
        "expected_per_category": expected_per_category,
        "category_counts": category_counts,
        "unexpected_categories": unexpected_categories,
        "rendering_task_ids": rendering_rows,
        "task_type_count": len(task_types),
        "task_type_counts": dict(sorted(task_types.items())),
        "normalized_output_format_count": len(normalized_outputs),
        "normalized_output_formats": sorted(normalized_outputs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a frozen textual StructEval development manifest.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-rows", type=int, default=20)
    args = parser.parse_args()

    result = validate_rows(
        load_structeval_manifest_rows(args.source, args.manifest),
        expected_rows=args.expected_rows,
    )
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
