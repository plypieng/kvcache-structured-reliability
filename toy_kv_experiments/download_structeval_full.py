from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import requests


DATASET = "TIGER-Lab/StructEval"
CONFIG = "default"
SPLIT = "test"
PARQUET_URL = (
    "https://huggingface.co/datasets/TIGER-Lab/StructEval/resolve/"
    "refs%2Fconvert%2Fparquet/default/test/0000.parquet"
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return None
    return value


def compact_preview(row: dict) -> dict:
    return {
        "task_id": row.get("task_id"),
        "task_name": row.get("task_name"),
        "input_type": row.get("input_type"),
        "output_type": row.get("output_type"),
        "rendering": row.get("rendering"),
        "query_preview": str(row.get("query", ""))[:700],
        "raw_output_metric": to_jsonable(row.get("raw_output_metric")),
        "vqa_count": len(to_jsonable(row.get("VQA")) or []),
    }


def main() -> None:
    out_dir = Path("toy_kv_experiments/data/structeval_full")
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "structeval_test.parquet"
    if not parquet_path.exists():
        response = requests.get(PARQUET_URL, timeout=60)
        response.raise_for_status()
        parquet_path.write_bytes(response.content)

    df = pd.read_parquet(parquet_path)
    rows = [to_jsonable(row) for row in df.to_dict(orient="records")]

    jsonl_path = out_dir / "structeval_test.jsonl"
    csv_path = out_dir / "structeval_test.csv"
    preview_path = out_dir / "preview_first_20.json"
    examples_path = out_dir / "examples_by_output_type.md"
    summary_path = out_dir / "summary.json"
    readme_path = out_dir / "README.md"

    write_jsonl(jsonl_path, rows)
    df.to_csv(csv_path, index=False)

    preview = [compact_preview(row) for row in rows[:20]]
    preview_path.write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")

    examples_seen: set[str] = set()
    example_lines = ["# StructEval Examples by Output Type", ""]
    for row in rows:
        output_type = str(row.get("output_type"))
        if output_type in examples_seen:
            continue
        examples_seen.add(output_type)
        example_lines.extend(
            [
                f"## {output_type}",
                "",
                f"- `task_id`: `{row.get('task_id')}`",
                f"- `task_name`: `{row.get('task_name')}`",
                f"- `input_type`: `{row.get('input_type')}`",
                f"- `rendering`: `{row.get('rendering')}`",
                "",
                "### Query Preview",
                "",
                "```text",
                str(row.get("query", ""))[:1600],
                "```",
                "",
                "### Raw Output Metric",
                "",
                "```json",
                json.dumps(to_jsonable(row.get("raw_output_metric")), ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    examples_path.write_text("\n".join(example_lines), encoding="utf-8")

    input_counts = Counter(str(row.get("input_type")) for row in rows)
    output_counts = Counter(str(row.get("output_type")) for row in rows)
    task_counts = Counter(str(row.get("task_name")) for row in rows)
    rendering_counts = Counter(str(row.get("rendering")) for row in rows)

    query_lengths = [len(str(row.get("query", ""))) for row in rows]
    vqa_counts = [len(row.get("VQA") or []) for row in rows]

    summary = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "num_rows": len(rows),
        "columns": list(df.columns),
        "files": {
            "parquet": str(parquet_path),
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "preview": str(preview_path),
            "examples_by_output_type": str(examples_path),
        },
        "input_type_counts": dict(input_counts.most_common()),
        "output_type_counts": dict(output_counts.most_common()),
        "task_name_top_20": dict(task_counts.most_common(20)),
        "rendering_counts": dict(rendering_counts.most_common()),
        "query_length": {
            "min": min(query_lengths),
            "max": max(query_lengths),
            "mean": sum(query_lengths) / len(query_lengths),
        },
        "vqa_count": {
            "min": min(vqa_counts),
            "max": max(vqa_counts),
            "mean": sum(vqa_counts) / len(vqa_counts),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    readme_path.write_text(
        "\n".join(
            [
                "# StructEval Full Local Copy",
                "",
                f"- Dataset: `{DATASET}`",
                f"- Config: `{CONFIG}`",
                f"- Split: `{SPLIT}`",
                f"- Rows: `{len(rows)}`",
                "",
                "## Files",
                "",
                "- `structeval_test.parquet`: original Hugging Face converted parquet shard",
                "- `structeval_test.jsonl`: full dataset as JSONL",
                "- `structeval_test.csv`: full dataset as CSV",
                "- `preview_first_20.json`: compact first-20-row preview",
                "- `examples_by_output_type.md`: one compact example per output type",
                "- `summary.json`: counts by input/output/task type",
                "",
                "## Columns",
                "",
                *[f"- `{column}`" for column in df.columns],
                "",
                "## Research Use",
                "",
                "Use this as an evaluation benchmark for structured-output generation, not as the main toy training dataset.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"downloaded StructEval to {out_dir}")
    print(f"rows: {len(rows)}")
    print("output_type counts:")
    for key, value in output_counts.most_common():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
