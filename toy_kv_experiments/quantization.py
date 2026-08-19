from __future__ import annotations

import torch


def fake_quantize_symmetric(
    x: torch.Tensor,
    bits: int = 8,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Simulate low-bit quantization error while returning a float tensor.

    This does not reduce memory. It performs quantize -> dequantize so the
    downstream attention code can remain ordinary PyTorch.
    """
    if bits >= 16:
        return x
    if bits < 2:
        raise ValueError("bits must be >= 2")

    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=dim, keepdim=True).clamp_min(eps) / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return q * scale


def quantize_kv_for_attention(
    k: torch.Tensor,
    v: torch.Tensor,
    bits: int = 8,
    key_axis: str = "per-channel",
    value_axis: str = "per-token",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake-quantize cached K,V tensors for attention.

    K,V shape: [batch, heads, tokens, head_dim].

    Axis meanings:
    - per-token: each token vector gets its own scale over head_dim.
    - per-channel: each channel gets its own scale over token positions.
    """
    k_dim = -2 if key_axis == "per-channel" else -1
    v_dim = -2 if value_axis == "per-channel" else -1
    return (
        fake_quantize_symmetric(k, bits=bits, dim=k_dim),
        fake_quantize_symmetric(v, bits=bits, dim=v_dim),
    )


def estimate_kv_cache_bytes(
    n_layer: int,
    n_head: int,
    seq_len: int,
    head_dim: int,
    bytes_per_value: float,
    batch_size: int = 1,
) -> float:
    """Estimate KV-cache size.

    Formula:
    batch * layers * 2(K,V) * heads * tokens * head_dim * bytes_per_value
    """
    return batch_size * n_layer * 2 * n_head * seq_len * head_dim * bytes_per_value
