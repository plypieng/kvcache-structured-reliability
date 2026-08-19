from __future__ import annotations

import argparse

try:
    from .quantization import estimate_kv_cache_bytes
except ImportError:
    from quantization import estimate_kv_cache_bytes


def fmt_bytes(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    formats = {
        "FP32": 4.0,
        "FP16/BF16": 2.0,
        "INT8": 1.0,
        "INT4 packed": 0.5,
        "INT2 packed": 0.25,
    }
    for name, bytes_per_value in formats.items():
        n = estimate_kv_cache_bytes(
            n_layer=args.layers,
            n_head=args.heads,
            seq_len=args.seq_len,
            head_dim=args.head_dim,
            bytes_per_value=bytes_per_value,
            batch_size=args.batch_size,
        )
        print(f"{name:10s}: {fmt_bytes(n)}")


if __name__ == "__main__":
    main()
