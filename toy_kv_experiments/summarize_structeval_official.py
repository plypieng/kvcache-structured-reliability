from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


CATEGORY_ORDER = (
    "T-generation",
    "T-conversion",
    "V-generation",
    "V-conversion",
)

EXPECTED_CATEGORY_COUNTS = {
    "T-generation": 250,
    "T-conversion": 700,
    "V-generation": 650,
    "V-conversion": 435,
}


def structeval_official_category(row: dict[str, Any]) -> str:
    """Map one official StructEval row to the leaderboard category."""
    track = "V" if bool(row.get("rendering", False)) else "T"
    task_kind = "generation" if row.get("input_type") == "Text" else "conversion"
    return f"{track}-{task_kind}"


def summarize_official_evaluation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the four category means used by the StructEval leaderboard."""
    scores: dict[str, list[float]] = defaultdict(list)
    unscored_rows: list[str] = []

    for row in rows:
        category = structeval_official_category(row)
        score = row.get("final_eval_score")
        if score is None:
            unscored_rows.append(str(row.get("task_id", "unknown")))
            continue
        scores[category].append(float(score))

    categories: dict[str, dict[str, Any]] = {}
    category_means: list[float] = []
    for category in CATEGORY_ORDER:
        values = scores.get(category, [])
        mean_score = fmean(values) if values else None
        if mean_score is not None:
            category_means.append(mean_score)
        categories[category] = {
            "count": len(values),
            "expected_count": EXPECTED_CATEGORY_COUNTS[category],
            "score_fraction": round(mean_score, 6) if mean_score is not None else None,
            "score_percent": round(100 * mean_score, 2) if mean_score is not None else None,
        }

    complete_counts = all(
        categories[category]["count"] == EXPECTED_CATEGORY_COUNTS[category]
        for category in CATEGORY_ORDER
    )
    official_average = fmean(category_means) if len(category_means) == len(CATEGORY_ORDER) else None
    all_scores = [score for category in CATEGORY_ORDER for score in scores.get(category, [])]
    task_weighted_average = fmean(all_scores) if all_scores else None

    return {
        "dataset_rows": len(rows),
        "scored_rows": len(all_scores),
        "unscored_rows": len(unscored_rows),
        "unscored_task_ids": unscored_rows,
        "categories": categories,
        "official_unweighted_average_fraction": (
            round(official_average, 6) if official_average is not None else None
        ),
        "official_unweighted_average_percent": (
            round(100 * official_average, 2) if official_average is not None else None
        ),
        "task_weighted_average_fraction": (
            round(task_weighted_average, 6) if task_weighted_average is not None else None
        ),
        "task_weighted_average_percent": (
            round(100 * task_weighted_average, 2) if task_weighted_average is not None else None
        ),
        "official_scope_complete": complete_counts and not unscored_rows and len(rows) == 2035,
    }


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError("StructEval evaluation output must be a JSON array of task objects")
    return payload


def format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "| StructEval category | Tasks | Expected | Score |",
        "|---|---:|---:|---:|",
    ]
    for category in CATEGORY_ORDER:
        values = summary["categories"][category]
        score = values["score_percent"]
        score_text = "n/a" if score is None else f"{score:.2f}"
        lines.append(
            f"| {category} | {values['count']} | {values['expected_count']} | {score_text} |"
        )
    overall = summary["official_unweighted_average_percent"]
    lines.append(
        f"| **Official unweighted average** | {summary['scored_rows']} | 2035 | "
        f"{'n/a' if overall is None else f'{overall:.2f}'} |"
    )
    lines.append("")
    lines.append(f"Official full-scope complete: `{str(summary['official_scope_complete']).lower()}`")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize an official StructEval evaluation.json into leaderboard categories."
    )
    parser.add_argument("evaluation_json")
    parser.add_argument("--out", help="Optional path for the machine-readable summary JSON")
    args = parser.parse_args()

    summary = summarize_official_evaluation(load_rows(args.evaluation_json))
    print(format_markdown(summary))
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
