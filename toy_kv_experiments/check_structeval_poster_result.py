from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from toy_kv_experiments.check_structeval_poster_manifest import validate_rows
from toy_kv_experiments.pretrained_kv_quantization import load_structeval_manifest_rows
from toy_kv_experiments.summarize_structeval_official import load_rows


REQUIRED_FIELDS = (
    "task_id",
    "input_type",
    "output_type",
    "query",
    "generation",
    "rendering",
)


def validate_result(
    rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
    *,
    key_bits: int,
    value_bits: int,
) -> dict[str, Any]:
    scope = validate_rows(rows)
    expected_ids = [str(row.get("task_id")) for row in expected_rows]
    actual_ids = [str(row.get("task_id")) for row in rows]
    missing_fields = {
        field: [index for index, row in enumerate(rows) if field not in row]
        for field in REQUIRED_FIELDS
    }
    missing_fields = {field: indices for field, indices in missing_fields.items() if indices}
    bit_mismatches = [
        str(row.get("task_id"))
        for row in rows
        if row.get("key_bits") != key_bits or row.get("value_bits") != value_bits
    ]
    fingerprints = {str(row.get("run_fingerprint")) for row in rows}
    valid = (
        scope["valid"]
        and actual_ids == expected_ids
        and not missing_fields
        and not bit_mismatches
        and len(fingerprints) == 1
    )
    return {
        "valid": valid,
        "scope": scope,
        "exact_manifest_order": actual_ids == expected_ids,
        "missing_fields": missing_fields,
        "bit_mismatch_count": len(bit_mismatches),
        "bit_mismatch_task_ids": bit_mismatches[:20],
        "run_fingerprint_count": len(fingerprints),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one completed poster-pilot result.")
    parser.add_argument("result")
    parser.add_argument("--source", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--key-bits", required=True, type=int)
    parser.add_argument("--value-bits", required=True, type=int)
    args = parser.parse_args()

    result = validate_result(
        load_rows(Path(args.result)),
        load_structeval_manifest_rows(args.source, args.manifest),
        key_bits=args.key_bits,
        value_bits=args.value_bits,
    )
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
