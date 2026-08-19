from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

try:
    from .dataset_sources import DATASET_SOURCES
except ImportError:
    from dataset_sources import DATASET_SOURCES


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").replace(":", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="toy_kv_experiments/papers")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, meta in DATASET_SOURCES.items():
        paper = meta.get("paper")
        if not paper:
            continue
        url = paper["pdf"]
        path = out_dir / f"{key}_{paper['arxiv']}.pdf"
        if path.exists() and path.stat().st_size > 0:
            print(f"already exists {path}")
            continue
        print(f"downloading {url} -> {path}")
        try:
            urlretrieve(url, path)
        except URLError as exc:
            print(f"warning: failed to download {url}: {exc}")

    print(f"wrote papers to {out_dir}")


if __name__ == "__main__":
    main()
