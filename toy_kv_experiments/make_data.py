from __future__ import annotations

import argparse
import random
from pathlib import Path


NAMES = ["alice", "bob", "carol", "dave", "erin", "frank"]
CITIES = ["tokyo", "osaka", "nagaoka", "kyoto", "sapporo"]
CROPS = ["tomato", "rice", "cucumber", "lettuce", "strawberry"]
TASKS = ["watering", "fertilizing", "harvesting", "seeding", "inspection"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]


def make_record() -> str:
    kind = random.choice(["person", "farm"])
    if kind == "person":
        return (
            '{"name":"'
            + random.choice(NAMES)
            + '","age":'
            + str(random.randint(18, 70))
            + ',"city":"'
            + random.choice(CITIES)
            + '"}'
        )
    return (
        '{"crop":"'
        + random.choice(CROPS)
        + '","task":"'
        + random.choice(TASKS)
        + '","day":"'
        + random.choice(DAYS)
        + '"}'
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="toy_kv_experiments/data/structured.txt")
    parser.add_argument("--records", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)
    lines = [make_record() for _ in range(args.records)]
    text = "\n".join(lines) + "\n"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out} ({len(text)} characters, {args.records} records)")


if __name__ == "__main__":
    main()
