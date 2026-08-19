from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import TinyGPTConfig
    from .quantization import quantize_kv_for_attention
except ImportError:  # Allows direct script execution from this folder.
    from config import TinyGPTConfig
    from quantization import quantize_kv_for_attention


PastKV = tuple[torch.Tensor, torch.Tensor]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.config = config
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: PastKV | None = None,
        use_cache: bool = False,
        quantize_cache: bool = False,
        bits: int = 8,
        key_axis: str = "per-channel",
        value_axis: str = "per-token",
        print_shapes: bool = False,
    ) -> tuple[torch.Tensor, PastKV | None]:
        batch, tokens, channels = x.shape

        # Project hidden states into Q,K,V.
        # q,k,v before reshape: [batch, tokens, n_embd]
        q, k, v = self.c_attn(x).split(self.config.n_embd, dim=2)

        # Split embedding dimension into heads.
        # q,k,v after reshape: [batch, heads, tokens, head_dim]
        q = q.view(batch, tokens, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, tokens, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, tokens, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            # Append new K,V to the cached token dimension.
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Educational quantization insertion point:
        # K,V have already been computed. Now the cached tensors are approximated.
        if quantize_cache:
            k_for_attn, v_for_attn = quantize_kv_for_attention(
                k,
                v,
                bits=bits,
                key_axis=key_axis,
                value_axis=value_axis,
            )
        else:
            k_for_attn, v_for_attn = k, v

        # Return what the next decoding step will reuse.
        # With fake quantization enabled, this simulates storing approximate K,V.
        new_kv = (k_for_attn, v_for_attn) if use_cache else None

        if print_shapes:
            print("q", tuple(q.shape), "k_cache", tuple(k.shape), "v_cache", tuple(v.shape))

        att = (q @ k_for_attn.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Training/prefill without cache needs a causal mask.
        # Cached decoding usually has tokens=1 and can attend to all cached positions.
        if past_kv is None and tokens > 1:
            att = att.masked_fill(self.tril[:, :, :tokens, :tokens] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        y = att @ v_for_attn
        y = y.transpose(1, 2).contiguous().view(batch, tokens, channels)
        return self.c_proj(y), new_kv


class Block(nn.Module):
    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: PastKV | None = None,
        use_cache: bool = False,
        quantize_cache: bool = False,
        bits: int = 8,
        key_axis: str = "per-channel",
        value_axis: str = "per-token",
        print_shapes: bool = False,
    ) -> tuple[torch.Tensor, PastKV | None]:
        attn_out, new_kv = self.attn(
            self.ln1(x),
            past_kv=past_kv,
            use_cache=use_cache,
            quantize_cache=quantize_cache,
            bits=bits,
            key_axis=key_axis,
            value_axis=value_axis,
            print_shapes=print_shapes,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv


class TinyGPT(nn.Module):
    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_kvs: list[PastKV | None] | None = None,
        use_cache: bool = False,
        quantize_cache: bool = False,
        bits: int = 8,
        key_axis: str = "per-channel",
        value_axis: str = "per-token",
        print_shapes: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[PastKV | None]]:
        batch, tokens = idx.shape
        if tokens > self.config.block_size:
            raise ValueError("sequence is longer than block_size")

        if past_kvs is None:
            past_len = 0
            past_kvs = [None] * len(self.blocks)
        else:
            first = past_kvs[0]
            past_len = 0 if first is None else first[0].shape[2]

        if past_len + tokens > self.config.block_size:
            raise ValueError("past_len + tokens exceeds block_size")

        pos = torch.arange(past_len, past_len + tokens, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)[None, :, :]

        new_past: list[PastKV | None] = []
        for block, past_kv in zip(self.blocks, past_kvs):
            x, new_kv = block(
                x,
                past_kv=past_kv,
                use_cache=use_cache,
                quantize_cache=quantize_cache,
                bits=bits,
                key_axis=key_axis,
                value_axis=value_axis,
                print_shapes=print_shapes,
            )
            new_past.append(new_kv)

        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
        return logits, loss, new_past
