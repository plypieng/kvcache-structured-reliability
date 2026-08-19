#!/usr/bin/env python3
"""Validate and summarize a paired FP16/KIVI shared-prefix trace."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_trace(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("task_traces"), list):
        raise ValueError(f"{path} is not a shared-prefix trace")
    return payload


def validate_pair(fp16: dict[str, Any], candidate: dict[str, Any]) -> None:
    if fp16.get("condition") != "fp16":
        raise ValueError("reference trace must use the fp16 condition")
    if candidate.get("condition") != "kivi":
        raise ValueError("candidate trace must use the kivi condition")
    for field in (
        "method",
        "model_name_or_path",
        "model_revision",
        "kivi_commit",
        "seed",
        "selected_task_ids",
    ):
        if fp16.get(field) != candidate.get(field):
            raise ValueError(f"trace mismatch in {field}")

    fp16_tasks = {str(task["task_id"]): task for task in fp16["task_traces"]}
    candidate_tasks = {str(task["task_id"]): task for task in candidate["task_traces"]}
    if list(fp16_tasks) != list(candidate_tasks):
        raise ValueError("trace task IDs or order do not match")
    for task_id, reference in fp16_tasks.items():
        compressed = candidate_tasks[task_id]
        for field in ("input_type", "output_type", "prompt_tokens", "frozen_token_ids"):
            if reference.get(field) != compressed.get(field):
                raise ValueError(f"task {task_id} differs in {field}")
        if len(reference.get("positions", [])) != len(compressed.get("positions", [])):
            raise ValueError(f"task {task_id} has a different number of positions")


def analyze(
    fp16: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    validate_pair(fp16, candidate)
    fp16_tasks = {str(task["task_id"]): task for task in fp16["task_traces"]}
    task_rows: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, Any]] = []

    for compressed in candidate["task_traces"]:
        task_id = str(compressed["task_id"])
        reference = fp16_tasks[task_id]
        first_disagreement = None
        for ref_position, position in zip(
            reference["positions"], compressed["positions"], strict=True
        ):
            if ref_position["top1_token_id"] == position["top1_token_id"]:
                continue
            if first_disagreement is None:
                first_disagreement = int(position["prediction_index"])
            disagreement_rows.append(
                {
                    "task_id": task_id,
                    "task_name": compressed["task_name"],
                    "input_type": compressed["input_type"],
                    "output_type": compressed["output_type"],
                    "prediction_index": int(position["prediction_index"]),
                    "target_token": position["target_token"],
                    "target_token_kind": position["target_token_kind"],
                    "fp16_top1_token": ref_position["top1_token"],
                    "kivi_top1_token": position["top1_token"],
                    "fp16_target_log_probability": ref_position[
                        "target_log_probability"
                    ],
                    "kivi_target_log_probability": position[
                        "target_log_probability"
                    ],
                    "target_log_probability_delta": (
                        position["target_log_probability"]
                        - ref_position["target_log_probability"]
                    ),
                }
            )

        task_rows.append(
            {
                "task_id": task_id,
                "task_name": compressed["task_name"],
                "input_type": compressed["input_type"],
                "output_type": compressed["output_type"],
                "positions": len(compressed["positions"]),
                "top1_disagreements": compressed["top1_disagreements_from_fp16"],
                "top1_disagreement_rate": compressed["top1_disagreement_rate"],
                "structural_positions": compressed["structural_positions"],
                "structural_top1_disagreements": compressed[
                    "structural_top1_disagreements"
                ],
                "structural_top1_disagreement_rate": compressed[
                    "structural_top1_disagreement_rate"
                ],
                "other_positions": compressed["other_positions"],
                "other_top1_disagreements": compressed["other_top1_disagreements"],
                "other_top1_disagreement_rate": compressed[
                    "other_top1_disagreement_rate"
                ],
                "first_top1_disagreement_index": first_disagreement,
            }
        )

    structural = [
        row
        for row in disagreement_rows
        if row["target_token_kind"] in {"structural_marker", "control_marker"}
    ]
    positions = sum(row["positions"] for row in task_rows)
    structural_positions = sum(row["structural_positions"] for row in task_rows)
    other_positions = sum(row["other_positions"] for row in task_rows)
    fp16_positions = [
        position
        for task in fp16["task_traces"]
        for position in task["positions"]
    ]
    candidate_positions = [
        position
        for task in candidate["task_traces"]
        for position in task["positions"]
    ]
    report = {
        "method": candidate["method"],
        "causal_boundary": candidate["causal_boundary"],
        "condition": f"K{candidate['k_bits']}/V{candidate['v_bits']}",
        "tasks": len(task_rows),
        "positions": positions,
        "tasks_with_top1_disagreement": sum(
            row["top1_disagreements"] > 0 for row in task_rows
        ),
        "top1_disagreements": len(disagreement_rows),
        "top1_disagreement_rate": len(disagreement_rows) / positions if positions else 0.0,
        "fp16_frozen_target_top1_matches": sum(
            row["top1_token_id"] == row["target_token_id"] for row in fp16_positions
        ),
        "fp16_frozen_target_top1_match_rate": (
            sum(row["top1_token_id"] == row["target_token_id"] for row in fp16_positions)
            / len(fp16_positions)
            if fp16_positions
            else 0.0
        ),
        "candidate_frozen_target_top1_matches": sum(
            row["top1_token_id"] == row["target_token_id"]
            for row in candidate_positions
        ),
        "candidate_frozen_target_top1_match_rate": (
            sum(
                row["top1_token_id"] == row["target_token_id"]
                for row in candidate_positions
            )
            / len(candidate_positions)
            if candidate_positions
            else 0.0
        ),
        "structural_positions": structural_positions,
        "structural_top1_disagreements": len(structural),
        "structural_top1_disagreement_rate": (
            len(structural) / structural_positions if structural_positions else 0.0
        ),
        "other_positions": other_positions,
        "other_top1_disagreements": len(disagreement_rows) - len(structural),
        "other_top1_disagreement_rate": (
            (len(disagreement_rows) - len(structural)) / other_positions
            if other_positions
            else 0.0
        ),
        "disagreements_by_target_kind": dict(
            Counter(row["target_token_kind"] for row in disagreement_rows)
        ),
    }
    return report, task_rows, disagreement_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Shared-prefix next-token sensitivity",
        "",
        "FP16 and KIVI receive the same prompt and the same previously generated",
        "FP16 tokens. A disagreement therefore occurs before free-running outputs",
        "can diverge, although it does not identify a causal cache entry or layer.",
        "",
        "| Measure | Result |",
        "|---|---:|",
        f"| Selected parse-regression tasks | {report['tasks']} |",
        f"| Teacher-forced positions | {report['positions']} |",
        f"| Tasks with a top-1 disagreement | {report['tasks_with_top1_disagreement']} |",
        f"| FP16 top-1 matches replay target | {report['fp16_frozen_target_top1_matches']} / {report['positions']} ({report['fp16_frozen_target_top1_match_rate']:.3%}) |",
        f"| KIVI top-1 matches replay target | {report['candidate_frozen_target_top1_matches']} / {report['positions']} ({report['candidate_frozen_target_top1_match_rate']:.3%}) |",
        f"| All top-1 disagreements | {report['top1_disagreements']} ({report['top1_disagreement_rate']:.3%}) |",
        f"| Structural-marker disagreements | {report['structural_top1_disagreements']} / {report['structural_positions']} ({report['structural_top1_disagreement_rate']:.3%}) |",
        f"| Other-token disagreements | {report['other_top1_disagreements']} / {report['other_positions']} ({report['other_top1_disagreement_rate']:.3%}) |",
        "",
        "## Interpretation boundary",
        "",
        report["causal_boundary"],
        "The evaluator artifact stores decoded output text rather than original token IDs.",
        "The trace retokenizes that text; any FP16 replay mismatch is reported above",
        "rather than silently treated as an effect of quantization.",
        "The selected tasks are KIVI-4-induced parse regressions, so these rates",
        "must not be generalized to all StructEval-T tasks.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp16", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, task_rows, disagreement_rows = analyze(
        load_trace(args.fp16),
        load_trace(args.candidate),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "task_summary.csv", task_rows)
    write_csv(args.output_dir / "top1_disagreements.csv", disagreement_rows)
    write_markdown(args.output_dir / "SUMMARY.md", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
