#!/usr/bin/env python3
"""Compare official KIVI LongBench reproduction results with the paper table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PAPER_FP16 = {
    "narrativeqa": 21.02,
    "qasper": 29.41,
    "multifieldqa_en": 47.13,
    "hotpotqa": 36.53,
    "musique": 19.13,
    "2wikimqa": 21.76,
    "gov_report": 32.59,
    "qmsum": 23.99,
    "multi_news": 27.09,
    "lcc": 53.49,
    "repobench-p": 51.40,
    "triviaqa": 86.23,
    "samsum": 43.04,
    "trec": 71.00,
    "passage_retrieval_en": 89.33,
}

PAPER_KIVI4 = {
    "narrativeqa": 20.97,
    "qasper": 29.41,
    "multifieldqa_en": 46.52,
    "hotpotqa": 36.25,
    "musique": 19.53,
    "2wikimqa": 21.66,
    "gov_report": 32.97,
    "qmsum": 24.06,
    "multi_news": 26.89,
    "lcc": 53.33,
    "repobench-p": 51.41,
    "triviaqa": 86.23,
    "samsum": 43.34,
    "trec": 71.00,
    "passage_retrieval_en": 89.42,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-dir", type=Path, required=True)
    parser.add_argument("--kivi4-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load_run(path: Path) -> tuple[dict[str, float], dict[str, Any]]:
    scores = json.loads((path / "result.json").read_text(encoding="utf-8"))
    metadata = json.loads((path / "run_metadata.json").read_text(encoding="utf-8"))
    return scores, metadata


def mean(values: dict[str, float]) -> float:
    return sum(values.values()) / len(values)


def main() -> None:
    args = parse_args()
    fp16, fp16_meta = load_run(args.fp16_dir)
    kivi4, kivi4_meta = load_run(args.kivi4_dir)
    expected_tasks = list(PAPER_FP16)

    missing_fp16 = sorted(set(expected_tasks) - set(fp16))
    missing_kivi4 = sorted(set(expected_tasks) - set(kivi4))
    common = [task for task in expected_tasks if task in fp16 and task in kivi4]
    if not common:
        raise RuntimeError("The two runs have no common published tasks.")

    rows = []
    for task in common:
        rows.append(
            {
                "task": task,
                "paper_fp16": PAPER_FP16[task],
                "our_fp16": fp16[task],
                "fp16_error": fp16[task] - PAPER_FP16[task],
                "paper_kivi4": PAPER_KIVI4[task],
                "our_kivi4": kivi4[task],
                "kivi4_error": kivi4[task] - PAPER_KIVI4[task],
                "paper_delta": PAPER_KIVI4[task] - PAPER_FP16[task],
                "our_delta": kivi4[task] - fp16[task],
            }
        )

    report = {
        "complete_15_task_reproduction": not missing_fp16 and not missing_kivi4,
        "missing_fp16": missing_fp16,
        "missing_kivi4": missing_kivi4,
        "common_task_count": len(common),
        "paper_fp16_mean_common": mean(
            {task: PAPER_FP16[task] for task in common}
        ),
        "our_fp16_mean_common": mean({task: fp16[task] for task in common}),
        "paper_kivi4_mean_common": mean(
            {task: PAPER_KIVI4[task] for task in common}
        ),
        "our_kivi4_mean_common": mean({task: kivi4[task] for task in common}),
        "paper_quality_delta_common": mean(
            {task: PAPER_KIVI4[task] - PAPER_FP16[task] for task in common}
        ),
        "our_quality_delta_common": mean(
            {task: kivi4[task] - fp16[task] for task in common}
        ),
        "fp16_metadata": fp16_meta,
        "kivi4_metadata": kivi4_meta,
        "tasks": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Official KIVI LongBench Reproduction",
        "",
        f"- Complete 15-task reproduction: `{report['complete_15_task_reproduction']}`",
        f"- Common tasks: `{len(common)}/15`",
        (
            "- FP16 mean on common tasks: "
            f"paper `{report['paper_fp16_mean_common']:.2f}`, "
            f"ours `{report['our_fp16_mean_common']:.2f}`"
        ),
        (
            "- KIVI-4 mean on common tasks: "
            f"paper `{report['paper_kivi4_mean_common']:.2f}`, "
            f"ours `{report['our_kivi4_mean_common']:.2f}`"
        ),
        (
            "- KIVI-4 minus FP16: "
            f"paper `{report['paper_quality_delta_common']:+.2f}`, "
            f"ours `{report['our_quality_delta_common']:+.2f}`"
        ),
        "",
        "| Task | Paper FP16 | Our FP16 | Paper KIVI-4 | Our KIVI-4 | Our delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['paper_fp16']:.2f} | "
            f"{row['our_fp16']:.2f} | {row['paper_kivi4']:.2f} | "
            f"{row['our_kivi4']:.2f} | {row['our_delta']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "A partial or limited run is a smoke test, not a paper reproduction.",
            "Differences must be interpreted with the recorded model, dataset,",
            "prompt, library, and KIVI commit metadata.",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
