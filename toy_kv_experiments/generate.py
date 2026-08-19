from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from .config import TinyGPTConfig
    from .data_utils import CharTokenizer
    from .model import TinyGPT
except ImportError:
    from config import TinyGPTConfig
    from data_utils import CharTokenizer
    from model import TinyGPT


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = TinyGPTConfig(**ckpt["config"])
    model = TinyGPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tokenizer = CharTokenizer.load(Path(checkpoint_path).with_suffix(".tokenizer.json"))
    return model, tokenizer, config


@torch.no_grad()
def generate_no_cache(model, tokenizer, prompt, max_new_tokens, temperature, device):
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _, _ = model(idx_cond)
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return tokenizer.decode(idx[0].detach().cpu().tolist())


@torch.no_grad()
def generate_with_cache(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    temperature,
    device,
    quantize_cache=False,
    bits=8,
    key_axis="per-channel",
    value_axis="per-token",
    print_shapes=False,
):
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    logits, _, past_kvs = model(
        prompt_ids,
        use_cache=True,
        quantize_cache=quantize_cache,
        bits=bits,
        key_axis=key_axis,
        value_axis=value_axis,
        print_shapes=print_shapes,
    )

    all_ids = prompt_ids.clone()
    next_logits = logits[:, -1, :]
    for _ in range(max_new_tokens):
        probs = F.softmax(next_logits / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        all_ids = torch.cat([all_ids, next_id], dim=1)
        if all_ids.shape[1] >= model.config.block_size:
            break
        logits, _, past_kvs = model(
            next_id,
            past_kvs=past_kvs,
            use_cache=True,
            quantize_cache=quantize_cache,
            bits=bits,
            key_axis=key_axis,
            value_axis=value_axis,
        )
        next_logits = logits[:, -1, :]
    return tokenizer.decode(all_ids[0].detach().cpu().tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="toy_kv_experiments/checkpoints/tiny_gpt.pt")
    parser.add_argument("--prompt", default='{"name":')
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--mode", choices=["no-cache", "cache"], default="cache")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--key-axis", choices=["per-token", "per-channel"], default="per-channel")
    parser.add_argument("--value-axis", choices=["per-token", "per-channel"], default="per-token")
    parser.add_argument("--print-shapes", action="store_true")
    args = parser.parse_args()

    device = pick_device()
    model, tokenizer, _ = load_model(args.checkpoint, device)

    if args.mode == "no-cache":
        text = generate_no_cache(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, device)
    else:
        text = generate_with_cache(
            model,
            tokenizer,
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            device,
            quantize_cache=args.quantize,
            bits=args.bits,
            key_axis=args.key_axis,
            value_axis=args.value_axis,
            print_shapes=args.print_shapes,
        )
    print(text)


if __name__ == "__main__":
    main()
