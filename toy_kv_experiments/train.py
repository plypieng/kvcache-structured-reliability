from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from .config import TinyGPTConfig
    from .data_utils import CharTokenizer, get_batch, load_dataset, split_data
    from .model import TinyGPT
except ImportError:
    from config import TinyGPTConfig
    from data_utils import CharTokenizer, get_batch, load_dataset, split_data
    from model import TinyGPT


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters=20):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(data, block_size, batch_size, device)
            _, loss, _ = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="toy_kv_experiments/data/structured.txt")
    parser.add_argument("--out", default="toy_kv_experiments/checkpoints/tiny_gpt.pt")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-interval", type=int, default=100)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"missing dataset: {data_path}. Run make_data.py first.")

    _, tokenizer, data = load_dataset(data_path)
    train_data, val_data = split_data(data)
    device = pick_device()

    config = TinyGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = TinyGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"device={device} vocab={tokenizer.vocab_size} parameters={sum(p.numel() for p in model.parameters())}")
    for step in range(args.max_steps + 1):
        if step % args.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, args.block_size, args.batch_size, device)
            print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")

        xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss, _ = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config.__dict__}, out)
    tokenizer.save(out.with_suffix(".tokenizer.json"))
    print(f"saved {out}")
    print(f"saved {out.with_suffix('.tokenizer.json')}")


if __name__ == "__main__":
    main()
