"""Freeze the matched failure/control tasks for the first UEP pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from toy_kv_experiments.summarize_structeval_official import structeval_official_category


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trace_task_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(task_id) for task_id in payload["selected_task_ids"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--failure-trace", required=True, type=Path)
    parser.add_argument("--control-trace", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    source_rows = {
        str(row.get("task_id")): row
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    failure_ids = load_trace_task_ids(args.failure_trace)
    control_ids = load_trace_task_ids(args.control_trace)
    if len(set(failure_ids)) != len(failure_ids) or len(set(control_ids)) != len(control_ids):
        raise ValueError("trace task IDs must be unique")
    overlap = set(failure_ids) & set(control_ids)
    if overlap:
        raise ValueError(f"failure/control task overlap: {sorted(overlap)}")

    tasks: list[dict[str, Any]] = []
    for role, task_ids in (("kivi4_failure", failure_ids), ("matched_control", control_ids)):
        for selection_index, task_id in enumerate(task_ids):
            row = source_rows.get(task_id)
            if row is None:
                raise ValueError(f"task {task_id} is not present in {source}")
            tasks.append(
                {
                    "selection_index": len(tasks),
                    "task_id": task_id,
                    "task_name": row.get("task_name"),
                    "input_type": row.get("input_type"),
                    "output_type": row.get("output_type"),
                    "stratum": f"{row.get('input_type')}->{row.get('output_type')}",
                    "leaderboard_category": structeval_official_category(row),
                    "pilot_role": role,
                    "source_trace_index": selection_index,
                }
            )

    manifest = {
        "manifest_version": 1,
        "manifest_type": "matched-budget-protection-pilot",
        "source_file": source.name,
        "source_sha256": sha256(source),
        "sampling": "shared-prefix-failure-control",
        "seed": 42,
        "task_count": len(tasks),
        "failure_task_count": len(failure_ids),
        "control_task_count": len(control_ids),
        "conditions": [
            "uniform_kivi4",
            "random_protection",
            "recency_protection",
            "structure_aware_protection",
        ],
        "budget_policy": {
            "cache_quantization_mode": "real-blockwise",
            "base_key_bits": 4,
            "base_value_bits": 4,
            "protected_bits": 8,
            "residual_length": 128,
            "group_size": 32,
            "target_average_bits_ceiling": 10.5,
            "protection_target": "both",
        },
        "tasks": tasks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(tasks)} tasks to {args.out}")
    print(f"failure tasks: {len(failure_ids)}")
    print(f"control tasks: {len(control_ids)}")


if __name__ == "__main__":
    main()
