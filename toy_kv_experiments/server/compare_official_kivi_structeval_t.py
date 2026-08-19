#!/usr/bin/env python3
"""Compare paired FP16, official KIVI-4, and official KIVI-2 StructEval-T outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from toy_kv_experiments.summarize_structeval_t import summarize


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array")
    return payload


def paired_summary(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_ids = [str(row.get("task_id")) for row in baseline]
    candidate_ids = [str(row.get("task_id")) for row in candidate]
    if candidate_ids != baseline_ids:
        raise ValueError("candidate task order does not match FP16")

    deltas = []
    regressions = []
    improvements = []
    new_parse_failures = []
    parse_recoveries = []
    path_regressions = []
    for fp16, compressed in zip(baseline, candidate, strict=True):
        task_id = str(fp16.get("task_id"))
        fp16_score = float(fp16.get("final_eval_score", 0))
        compressed_score = float(compressed.get("final_eval_score", 0))
        delta = compressed_score - fp16_score
        deltas.append(delta)
        detail = {
            "task_id": task_id,
            "input_type": fp16.get("input_type"),
            "output_type": fp16.get("output_type"),
            "fp16_score": fp16_score,
            "candidate_score": compressed_score,
            "delta": round(delta, 6),
        }
        if delta < 0:
            regressions.append(detail)
        elif delta > 0:
            improvements.append(detail)
        if float(fp16.get("render_score", 0)) == 1 and float(compressed.get("render_score", 0)) == 0:
            new_parse_failures.append(detail)
        if float(fp16.get("render_score", 0)) == 0 and float(compressed.get("render_score", 0)) == 1:
            parse_recoveries.append(detail)
        if float(compressed.get("key_validation_score", 0)) < float(fp16.get("key_validation_score", 0)):
            path_regressions.append(detail)

    return {
        "mean_task_delta": round(fmean(deltas), 6),
        "regressed_tasks": len(regressions),
        "improved_tasks": len(improvements),
        "unchanged_tasks": len(deltas) - len(regressions) - len(improvements),
        "new_parse_failures": len(new_parse_failures),
        "parse_recoveries": len(parse_recoveries),
        "required_path_regressions": len(path_regressions),
        "largest_regressions": sorted(regressions, key=lambda row: row["delta"])[:20],
        "new_parse_failure_examples": new_parse_failures[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp16", type=Path, required=True)
    parser.add_argument("--kivi4", type=Path, required=True)
    parser.add_argument("--kivi2", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    fp16 = load_rows(args.fp16)
    kivi4 = load_rows(args.kivi4)
    kivi2 = load_rows(args.kivi2)
    result = {
        "scope": "complete StructEval-T paired comparison",
        "conditions": {
            "FP16": summarize(fp16),
            "KIVI-4": summarize(kivi4),
            "KIVI-2": summarize(kivi2),
        },
        "paired": {
            "KIVI-4_vs_FP16": paired_summary(fp16, kivi4),
            "KIVI-2_vs_FP16": paired_summary(fp16, kivi2),
        },
    }
    args.output_json.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Official KIVI on StructEval-T",
        "",
        "| Condition | T-generation | T-conversion | T mean | Parse success |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in ("FP16", "KIVI-4", "KIVI-2"):
        summary = result["conditions"][condition]
        generation = summary["categories"]["T-generation"]
        conversion = summary["categories"]["T-conversion"]
        parse_success = (
            generation["parse_success_count"] + conversion["parse_success_count"]
        )
        lines.append(
            f"| {condition} | {generation['mean_score']:.4f} | "
            f"{conversion['mean_score']:.4f} | "
            f"{summary['t_track_unweighted_mean']:.4f} | "
            f"{parse_success}/950 |"
        )
    lines.extend(
        [
            "",
            "| Pair | Regressed | Improved | New parse failures | Path regressions |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for pair in ("KIVI-4_vs_FP16", "KIVI-2_vs_FP16"):
        paired = result["paired"][pair]
        lines.append(
            f"| {pair.replace('_', ' ')} | {paired['regressed_tasks']} | "
            f"{paired['improved_tasks']} | {paired['new_parse_failures']} | "
            f"{paired['required_path_regressions']} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
