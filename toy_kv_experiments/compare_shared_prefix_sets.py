#!/usr/bin/env python3
"""Compare shared-prefix sensitivity summaries for failures and controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not an analysis summary")
    return payload


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def compare(failure: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "structural_top1_disagreement_rate",
        "other_top1_disagreement_rate",
        "top1_disagreement_rate",
    )
    return {
        "failure_tasks": failure["tasks"],
        "control_tasks": control["tasks"],
        "failure_positions": failure["positions"],
        "control_positions": control["positions"],
        "failure": {field: failure[field] for field in fields},
        "control": {field: control[field] for field in fields},
        "failure_minus_control": {
            field: failure[field] - control[field] for field in fields
        },
        "failure_over_control": {
            field: ratio(failure[field], control[field]) for field in fields
        },
        "interpretation_boundary": (
            "The control set is matched by input/output format and FP16 generated length, "
            "but it is not a randomized or semantic match. The comparison is descriptive "
            "and does not establish that structural positions cause parse failures."
        ),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    failure = report["failure"]
    control = report["control"]
    delta = report["failure_minus_control"]
    ratio_values = report["failure_over_control"]
    lines = [
        "# Shared-prefix failure-set versus control-set comparison",
        "",
        f"The two sets contain {report['failure_tasks']} tasks each and are matched by",
        "input/output format and nearest FP16 generated length.",
        "",
        "| Measure | KIVI-4-induced parse failures | Matched non-regression controls | Difference | Ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = (
        ("top1_disagreement_rate", "All top-1 disagreement rate"),
        ("structural_top1_disagreement_rate", "Structural-marker disagreement rate"),
        ("other_top1_disagreement_rate", "Other-token disagreement rate"),
    )
    for field, label in labels:
        ratio_text = "n/a" if ratio_values[field] is None else f"{ratio_values[field]:.2f}x"
        lines.append(
            f"| {label} | {failure[field]:.3%} | {control[field]:.3%} | "
            f"{delta[field]:+.3%} | {ratio_text} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            report["interpretation_boundary"],
            "The shared-prefix trace identifies changed next-token decisions under an",
            "identical prefix; it does not identify a specific cache position, layer, Key,",
            "or Value as the cause.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failures", required=True, type=Path)
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare(load(args.failures), load(args.controls))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(args.output_dir / "COMPARISON.md", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
