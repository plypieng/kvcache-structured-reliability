from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def write_split(split: str, out_dir: Path, text_key: str = "text") -> None:
    txt_path = out_dir / f"{split}.txt"
    jsonl_path = out_dir / f"{split}.jsonl"
    done_path = out_dir / f"{split}.done"

    if done_path.exists() and txt_path.exists() and jsonl_path.exists():
        print(f"skip {split}: already complete")
        return

    tmp_txt = out_dir / f"{split}.txt.tmp"
    tmp_jsonl = out_dir / f"{split}.jsonl.tmp"
    tmp_txt.unlink(missing_ok=True)
    tmp_jsonl.unlink(missing_ok=True)

    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    count = 0
    char_count = 0

    with tmp_txt.open("w") as txt, tmp_jsonl.open("w") as jsonl:
        for row in ds:
            story = str(row.get(text_key, "")).strip()
            if not story:
                continue
            txt.write(story)
            txt.write("\n\n")
            jsonl.write(json.dumps({"text": story}, ensure_ascii=False))
            jsonl.write("\n")
            count += 1
            char_count += len(story)
            if count % 10000 == 0:
                print(f"{split}: {count:,} stories, {char_count:,} chars")

    tmp_txt.replace(txt_path)
    tmp_jsonl.replace(jsonl_path)
    done_path.write_text(f"stories={count}\nchars={char_count}\n")
    print(f"complete {split}: {count:,} stories, {char_count:,} chars")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="toy_kv_experiments/data/tinystories_full")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        write_split(split, out_dir)

    print(f"wrote TinyStories full data to {out_dir}")


if __name__ == "__main__":
    main()
