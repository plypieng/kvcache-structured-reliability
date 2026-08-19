from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from toy_kv_experiments.pretrained_kv_quantization import load_structeval_manifest_rows
from toy_kv_experiments.summarize_structeval_official import structeval_official_category


EXPECTED_CATEGORY_COUNTS = {
    "T-generation": 250,
    "T-conversion": 700,
}
EXPECTED_OUTPUT_TYPE_COUNTS = {
    "CSV": 200,
    "JSON": 250,
    "TOML": 50,
    "XML": 200,
    "YAML": 250,
}
EXPECTED_ROWS = sum(EXPECTED_CATEGORY_COUNTS.values())


def load_source_rows(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_rows(
    rows: list[dict[str, Any]],
    *,
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_rows = [row for row in source_rows if not bool(row.get("rendering", False))]
    expected_ids = [str(row.get("task_id") or "") for row in expected_rows]
    actual_ids = [str(row.get("task_id") or "") for row in rows]
    categories = Counter(structeval_official_category(row) for row in rows)
    outputs = Counter(str(row.get("output_type") or "") for row in rows)
    rendering_task_ids = [
        str(row.get("task_id")) for row in rows if bool(row.get("rendering", False))
    ]

    category_counts = {
        category: categories.get(category, 0)
        for category in EXPECTED_CATEGORY_COUNTS
    }
    unexpected_categories = {
        category: count
        for category, count in categories.items()
        if category not in EXPECTED_CATEGORY_COUNTS
    }
    output_type_counts = {
        output_type: outputs.get(output_type, 0)
        for output_type in EXPECTED_OUTPUT_TYPE_COUNTS
    }
    unexpected_output_types = {
        output_type: count
        for output_type, count in outputs.items()
        if output_type not in EXPECTED_OUTPUT_TYPE_COUNTS
    }

    valid = (
        len(rows) == EXPECTED_ROWS
        and len(set(actual_ids)) == EXPECTED_ROWS
        and all(actual_ids)
        and actual_ids == expected_ids
        and category_counts == EXPECTED_CATEGORY_COUNTS
        and not unexpected_categories
        and output_type_counts == EXPECTED_OUTPUT_TYPE_COUNTS
        and not unexpected_output_types
        and not rendering_task_ids
    )
    return {
        "valid": valid,
        "rows": len(rows),
        "expected_rows": EXPECTED_ROWS,
        "unique_task_ids": len(set(actual_ids)),
        "exact_source_order": actual_ids == expected_ids,
        "category_counts": category_counts,
        "expected_category_counts": EXPECTED_CATEGORY_COUNTS,
        "unexpected_categories": unexpected_categories,
        "output_type_counts": output_type_counts,
        "expected_output_type_counts": EXPECTED_OUTPUT_TYPE_COUNTS,
        "unexpected_output_types": unexpected_output_types,
        "rendering_task_ids": rendering_task_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a complete, non-rendered StructEval-T manifest."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    result = validate_rows(
        load_structeval_manifest_rows(args.source, args.manifest),
        source_rows=load_source_rows(args.source),
    )
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
