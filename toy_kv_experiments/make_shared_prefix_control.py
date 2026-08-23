#!/usr/bin/env python3
"""Select a deterministic non-regression control for shared-prefix analysis."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_controls(
    rows: list[dict[str, str]],
    *,
    condition: str = "KIVI-4",
    failure_transition: str = "fp16_only",
    control_transition: str = "both_pass",
) -> list[dict[str, str]]:
    scoped = [row for row in rows if row.get("condition") == condition]
    failures = [row for row in scoped if row.get("parse_transition") == failure_transition]
    controls = [row for row in scoped if row.get("parse_transition") == control_transition]
    if not failures:
        raise ValueError("no failure rows matched the requested condition")

    available: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in controls:
        key = (row.get("input_type", ""), row.get("output_type", ""))
        available.setdefault(key, []).append(row)
    for candidates in available.values():
        candidates.sort(key=lambda row: (int(row["fp16_generated_tokens"]), row["task_id"]))

    selected: list[dict[str, str]] = []
    for failure in sorted(failures, key=lambda row: row["task_id"]):
        key = (failure.get("input_type", ""), failure.get("output_type", ""))
        candidates = available.get(key, [])
        if not candidates:
            raise ValueError(f"no controls available for stratum {key}")
        target_length = int(failure["fp16_generated_tokens"])
        candidate = min(
            candidates,
            key=lambda row: (
                abs(int(row["fp16_generated_tokens"]) - target_length),
                row["task_id"],
            ),
        )
        candidates.remove(candidate)
        selected.append(
            {
                "condition": condition,
                "task_id": candidate["task_id"],
                "task_name": candidate["task_name"],
                "input_type": candidate["input_type"],
                "output_type": candidate["output_type"],
                "control_role": "matched_non_regression_control",
                "matched_failure_task_id": failure["task_id"],
                "failure_fp16_generated_tokens": failure["fp16_generated_tokens"],
                "control_fp16_generated_tokens": candidate["fp16_generated_tokens"],
                "length_delta": str(
                    int(candidate["fp16_generated_tokens"])
                    - int(failure["fp16_generated_tokens"])
                ),
                "failure_parse_transition": failure["parse_transition"],
                "control_parse_transition": candidate["parse_transition"],
            }
        )
    return selected


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = select_controls(load_rows(args.transitions))
    write_rows(args.output, rows)
    print(f"wrote {len(rows)} matched controls to {args.output}")


if __name__ == "__main__":
    main()
