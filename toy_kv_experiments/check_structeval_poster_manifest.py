from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from toy_kv_experiments.pretrained_kv_quantization import load_structeval_manifest_rows
from toy_kv_experiments.summarize_structeval_official import (
    CATEGORY_ORDER,
    structeval_official_category,
)


EXPECTED_CATEGORY_COUNT = 25
EXPECTED_ROWS = 100
EXPECTED_TASK_TYPES = 44
EXPECTED_OUTPUT_FORMATS = 18


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_ids = [str(row.get("task_id") or "") for row in rows]
    categories = Counter(structeval_official_category(row) for row in rows)
    task_types = Counter(f"{row.get('input_type')}->{row.get('output_type')}" for row in rows)
    normalized_outputs = {str(row.get("output_type") or "").lower() for row in rows}
    category_counts = {category: categories.get(category, 0) for category in CATEGORY_ORDER}
    valid = (
        len(rows) == EXPECTED_ROWS
        and len(set(task_ids)) == EXPECTED_ROWS
        and all(task_ids)
        and all(count == EXPECTED_CATEGORY_COUNT for count in category_counts.values())
        and len(task_types) == EXPECTED_TASK_TYPES
        and len(normalized_outputs) == EXPECTED_OUTPUT_FORMATS
    )
    return {
        "valid": valid,
        "rows": len(rows),
        "unique_task_ids": len(set(task_ids)),
        "category_counts": category_counts,
        "task_type_count": len(task_types),
        "task_type_counts": dict(sorted(task_types.items())),
        "normalized_output_format_count": len(normalized_outputs),
        "normalized_output_formats": sorted(normalized_outputs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the frozen 100-task all-format StructEval poster manifest."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    result = validate_rows(load_structeval_manifest_rows(args.source, args.manifest))
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
