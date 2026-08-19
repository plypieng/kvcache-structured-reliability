from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from toy_kv_experiments.check_structeval_official_artifact import validate_summary
from toy_kv_experiments.summarize_structeval_official import CATEGORY_ORDER


def load_summary(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"StructEval summary must be a JSON object: {path}")
    return payload


def build_matrix(conditions: list[tuple[str, str | Path]]) -> dict[str, Any]:
    rows = []
    for label, path in conditions:
        summary = load_summary(path)
        validation = validate_summary(summary)
        rows.append(
            {
                "condition": label,
                "summary_path": str(path),
                "official_scope_complete": validation["valid"],
                "categories": {
                    category: summary.get("categories", {})
                    .get(category, {})
                    .get("score_percent")
                    for category in CATEGORY_ORDER
                },
                "official_unweighted_average_percent": summary.get(
                    "official_unweighted_average_percent"
                ),
            }
        )
    return {
        "all_conditions_official_scope_complete": all(
            row["official_scope_complete"] for row in rows
        ),
        "conditions": rows,
    }


def format_markdown(matrix: dict[str, Any]) -> str:
    lines = [
        "| Condition | T-gen | T-conv | V-gen | V-conv | Official average | Complete |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in matrix["conditions"]:
        values = []
        for category in CATEGORY_ORDER:
            value = row["categories"][category]
            values.append("n/a" if value is None else f"{value:.2f}")
        overall = row["official_unweighted_average_percent"]
        lines.append(
            f"| {row['condition']} | {' | '.join(values)} | "
            f"{'n/a' if overall is None else f'{overall:.2f}'} | "
            f"{'yes' if row['official_scope_complete'] else 'no'} |"
        )
    lines.append("")
    lines.append(
        "All conditions complete: "
        f"`{str(matrix['all_conditions_official_scope_complete']).lower()}`"
    )
    return "\n".join(lines)


def parse_condition(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("condition must be LABEL=SUMMARY_PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("condition must be LABEL=SUMMARY_PATH")
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine completed official StructEval summaries into one matrix."
    )
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        type=parse_condition,
        metavar="LABEL=SUMMARY_PATH",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown-out")
    args = parser.parse_args()

    matrix = build_matrix(args.condition)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")

    markdown = format_markdown(matrix)
    print(markdown)
    if args.markdown_out:
        markdown_output = Path(args.markdown_out)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(markdown + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
