from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from toy_kv_experiments.summarize_structeval_official import structeval_official_category


EXPECTED_CATEGORY_COUNTS = {
    "T-generation": 250,
    "T-conversion": 700,
}
EXPECTED_ROWS = sum(EXPECTED_CATEGORY_COUNTS.values())


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("StructEval evaluation artifact must be a JSON array")
    return payload


def summarize(rows: list[dict[str, Any]], *, expected_rows: int = EXPECTED_ROWS) -> dict[str, Any]:
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    output_types: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unscored_task_ids = []
    rendered_task_ids = []

    for row in rows:
        if bool(row.get("rendering", False)):
            rendered_task_ids.append(str(row.get("task_id")))
        if row.get("final_eval_score") is None:
            unscored_task_ids.append(str(row.get("task_id")))
        categories[structeval_official_category(row)].append(row)
        output_types[str(row.get("output_type"))].append(row)

    def group_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
        scores = [float(row["final_eval_score"]) for row in group if row.get("final_eval_score") is not None]
        parse_scores = [float(row.get("render_score", 0)) for row in group]
        path_scores = [float(row.get("key_validation_score", 0)) for row in group]
        return {
            "count": len(group),
            "mean_score": round(fmean(scores), 6) if scores else None,
            "parse_success_count": sum(score == 1 for score in parse_scores),
            "parse_success_rate": round(fmean(parse_scores), 6) if parse_scores else None,
            "mean_required_path_score": round(fmean(path_scores), 6) if path_scores else None,
            "perfect_score_count": sum(score == 1 for score in scores),
        }

    category_summaries = {
        category: group_summary(categories.get(category, []))
        for category in EXPECTED_CATEGORY_COUNTS
    }
    category_means = [
        category_summaries[category]["mean_score"]
        for category in EXPECTED_CATEGORY_COUNTS
        if category_summaries[category]["mean_score"] is not None
    ]
    task_scores = [
        float(row["final_eval_score"])
        for row in rows
        if row.get("final_eval_score") is not None
    ]
    complete_scope = (
        expected_rows == EXPECTED_ROWS
        and len(rows) == EXPECTED_ROWS
        and all(
            category_summaries[category]["count"] == expected
            for category, expected in EXPECTED_CATEGORY_COUNTS.items()
        )
        and not unscored_task_ids
        and not rendered_task_ids
    )

    return {
        "scope": "StructEval-T",
        "complete_scope": complete_scope,
        "dataset_rows": len(rows),
        "expected_rows": expected_rows,
        "unscored_task_ids": unscored_task_ids,
        "rendered_task_ids": rendered_task_ids,
        "categories": category_summaries,
        "t_track_unweighted_mean": (
            round(fmean(category_means), 6)
            if len(category_means) == len(EXPECTED_CATEGORY_COUNTS)
            else None
        ),
        "task_weighted_mean": round(fmean(task_scores), 6) if task_scores else None,
        "by_output_type": {
            output_type: group_summary(group)
            for output_type, group in sorted(output_types.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the complete StructEval-T track.")
    parser.add_argument("evaluation")
    parser.add_argument("--out")
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    args = parser.parse_args()

    summary = summarize(load_rows(args.evaluation), expected_rows=args.expected_rows)
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.expected_rows == EXPECTED_ROWS and not summary["complete_scope"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
