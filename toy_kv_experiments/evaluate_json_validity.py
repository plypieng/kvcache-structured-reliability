from __future__ import annotations

import argparse
import json

from generate import generate_with_cache, load_model, pick_device


def is_json_prefix_completion_valid(text: str) -> bool:
    """Try to parse the first complete-looking JSON object in generated text."""
    start = text.find("{")
    end = text.find("}", start + 1)
    if start == -1 or end == -1:
        return False
    candidate = text[start : end + 1]
    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="toy_kv_experiments/checkpoints/tiny_gpt.pt")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--prompt", default="{")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    device = pick_device()
    model, tokenizer, _ = load_model(args.checkpoint, device)

    for quantize in [False, True]:
        ok = 0
        for _ in range(args.trials):
            text = generate_with_cache(
                model,
                tokenizer,
                args.prompt,
                args.max_new_tokens,
                temperature=0.8,
                device=device,
                quantize_cache=quantize,
                bits=args.bits,
            )
            ok += int(is_json_prefix_completion_valid(text))
        label = f"fake INT{args.bits} KV" if quantize else "full precision KV"
        print(f"{label:18s}: {ok}/{args.trials} valid")


if __name__ == "__main__":
    main()
