from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from toy_kv_experiments.summarize_structeval_official import (
    CATEGORY_ORDER,
    structeval_official_category,
)


EXPECTED_PILOT_COUNT = 25


def summarize_poster_evaluation(
    rows: list[dict[str, Any]],
    *,
    expected_category_count: int = EXPECTED_PILOT_COUNT,
) -> dict[str, Any]:
    scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        score = row.get("final_eval_score")
        if score is not None:
            scores[structeval_official_category(row)].append(float(score))

    categories = {}
    available_means = []
    for category in CATEGORY_ORDER:
        values = scores.get(category, [])
        mean_score = fmean(values) if values else None
        if mean_score is not None:
            available_means.append(mean_score)
        categories[category] = {
            "count": len(values),
            "expected_pilot_count": expected_category_count,
            "score_percent": round(100 * mean_score, 2) if mean_score is not None else None,
        }

    t_complete = (
        categories["T-generation"]["count"] == expected_category_count
        and categories["T-conversion"]["count"] == expected_category_count
    )
    full_complete = len(rows) == expected_category_count * len(CATEGORY_ORDER) and all(
        categories[category]["count"] == expected_category_count for category in CATEGORY_ORDER
    )
    t_means = [
        categories[category]["score_percent"]
        for category in ("T-generation", "T-conversion")
        if categories[category]["score_percent"] is not None
    ]
    return {
        "dataset_rows": len(rows),
        "categories": categories,
        "available_category_average_percent": (
            round(100 * fmean(available_means), 2) if available_means else None
        ),
        "t_category_average_percent": round(fmean(t_means), 2) if len(t_means) == 2 else None,
        "t_pilot_complete": t_complete and len(rows) == expected_category_count * 2,
        "full_pilot_complete": full_complete,
        "official_leaderboard_comparable": False,
    }


def format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "| Poster category | Tasks | Score |",
        "|---|---:|---:|",
    ]
    for category in CATEGORY_ORDER:
        values = summary["categories"][category]
        score = values["score_percent"]
        lines.append(
            f"| {category} | {values['count']} | {'n/a' if score is None else f'{score:.2f}'} |"
        )
    lines.extend(
        [
            "",
            f"StructEval-T pilot average: `{summary['t_category_average_percent']}`",
            f"T pilot complete: `{str(summary['t_pilot_complete']).lower()}`",
            "Official leaderboard comparable: `false`",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the frozen StructEval-100 poster panel.")
    parser.add_argument("evaluation_json")
    parser.add_argument("--out")
    parser.add_argument("--expected-category-count", type=int, default=EXPECTED_PILOT_COUNT)
    args = parser.parse_args()

    payload = json.loads(Path(args.evaluation_json).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError("poster evaluation must be a JSON array of task objects")
    summary = summarize_poster_evaluation(
        payload,
        expected_category_count=args.expected_category_count,
    )
    print(format_markdown(summary))
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
