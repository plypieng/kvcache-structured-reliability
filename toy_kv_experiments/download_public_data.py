from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import get_dataset_split_names, load_dataset
from huggingface_hub import hf_hub_download

try:
    from .dataset_sources import DATASET_SOURCES
except ImportError:
    from dataset_sources import DATASET_SOURCES


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_existing(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def download_tinystories(out_dir: Path, max_rows: int) -> None:
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    rows = []
    texts = []
    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        text = str(first_existing(row, ["text", "story"]) or "")
        rows.append({"text": text})
        texts.append(text.strip())

    write_jsonl(out_dir / "tinystories_sample.jsonl", rows)
    (out_dir / "tinystories_train.txt").write_text("\n\n".join(texts) + "\n")


def download_structeval(out_dir: Path, max_rows: int) -> None:
    split = "train" if "train" in get_dataset_split_names("TIGER-Lab/StructEval") else "test"
    ds = load_dataset("TIGER-Lab/StructEval", split=split)
    rows = []
    prompts = []
    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        row = dict(row)
        rows.append(row)

        prompt = first_existing(row, ["query", "prompt", "instruction", "question", "input"])
        target = first_existing(row, ["answer", "output", "target", "reference", "raw_output_metric"])
        if prompt is not None:
            prompts.append("PROMPT:\n" + str(prompt).strip())
        if target is not None:
            prompts.append("TARGET:\n" + str(target).strip())

    write_jsonl(out_dir / "structeval_sample.jsonl", rows)
    (out_dir / "structeval_text.txt").write_text("\n\n".join(prompts) + "\n")


def download_synthetic_structured(out_dir: Path, max_rows: int) -> None:
    rows = []
    rendered = []

    # The HF datasets builder currently fails for this repo because nested JSON
    # fields have inconsistent shapes. Reading the raw JSONL is more robust.
    raw_path = hf_hub_download(
        repo_id="mdonigian/synthetic-structured-output-dataset",
        repo_type="dataset",
        filename="sft_synthetic_json.jsonl",
    )
    with open(raw_path) as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            rows.append(json.loads(line))

    for row in rows:
        query = first_existing(row, ["query", "prompt", "instruction", "input"])
        chosen = first_existing(row, ["chosen", "output", "answer"])
        if query is not None:
            rendered.append("QUERY:\n" + str(query).strip())
        if chosen is not None:
            chosen_text = json.dumps(chosen, ensure_ascii=False) if isinstance(chosen, (dict, list)) else str(chosen).strip()
            rendered.append("CHOSEN:\n" + chosen_text)

    write_jsonl(out_dir / "synthetic_structured_sample.jsonl", rows)
    (out_dir / "synthetic_structured_text.txt").write_text("\n\n".join(rendered) + "\n")


def download_sob(out_dir: Path, max_rows: int) -> None:
    ds = load_dataset("interfaze-ai/sob", "default", split="train")
    rows = []
    rendered = []
    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        row = dict(row)
        rows.append(row)

        context = first_existing(row, ["context", "source", "document", "text", "input"])
        question = first_existing(row, ["question", "query", "prompt", "instruction"])
        schema = first_existing(row, ["schema", "json_schema", "output_schema"])
        answer = first_existing(row, ["answer", "ground_truth", "target", "output"])

        if context is not None:
            rendered.append("CONTEXT:\n" + str(context).strip())
        if question is not None:
            rendered.append("QUESTION:\n" + str(question).strip())
        if schema is not None:
            schema_text = json.dumps(schema, ensure_ascii=False) if isinstance(schema, (dict, list)) else str(schema).strip()
            rendered.append("SCHEMA:\n" + schema_text)
        if answer is not None:
            answer_text = json.dumps(answer, ensure_ascii=False) if isinstance(answer, (dict, list)) else str(answer).strip()
            rendered.append("ANSWER:\n" + answer_text)

    write_jsonl(out_dir / "sob_sample.jsonl", rows)
    (out_dir / "sob_text.txt").write_text("\n\n".join(rendered) + "\n")


def write_metadata(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_sources.json").write_text(json.dumps(DATASET_SOURCES, ensure_ascii=False, indent=2))

    lines = ["# Dataset Sources", ""]
    for name, meta in DATASET_SOURCES.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- Dataset: {meta.get('hf_id') or meta.get('dataset_url') or 'not downloaded by this script'}")
        lines.append(f"- Description: {meta['description']}")
        if meta.get("dataset_url"):
            lines.append(f"- Dataset URL: {meta['dataset_url']}")
        paper = meta.get("paper")
        if paper:
            lines.append(f"- Paper: {paper['title']}")
            lines.append(f"- arXiv: {paper['url']}")
            lines.append(f"- PDF: {paper['pdf']}")
        else:
            lines.append("- Paper: no corresponding paper found; dataset card only")
        lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="toy_kv_experiments/data/public")
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument(
        "--dataset",
        choices=["all", "tinystories", "structeval", "synthetic_structured", "sob"],
        default="all",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(out_dir)

    if args.dataset in ("all", "tinystories"):
        print("downloading TinyStories sample")
        download_tinystories(out_dir, args.max_rows)
    if args.dataset in ("all", "structeval"):
        print("downloading StructEval sample")
        download_structeval(out_dir, args.max_rows)
    if args.dataset in ("all", "synthetic_structured"):
        print("downloading synthetic structured-output sample")
        download_synthetic_structured(out_dir, args.max_rows)
    if args.dataset in ("all", "sob"):
        print("downloading SOB sample")
        download_sob(out_dir, args.max_rows)

    print(f"wrote data and metadata to {out_dir}")


if __name__ == "__main__":
    main()
