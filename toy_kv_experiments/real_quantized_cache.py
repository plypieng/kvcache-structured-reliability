from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


StorageLayout = Literal["keys", "values"]
SUPPORTED_STORAGE_BITS = frozenset({2, 4, 8, 16})
CACHE_ALGORITHM_REVISION = "kivi-post-attention-finalize-v3"


def _validate_storage_bits(bits: int) -> None:
    if bits not in SUPPORTED_STORAGE_BITS:
        raise ValueError(
            "real KIVI-style storage supports only 2, 4, 8, or 16 bits; "
            f"got {bits}. Use fake quantization for numerical-only sweeps."
        )


def _pack_unsigned(values: torch.Tensor, bits: int) -> tuple[torch.Tensor, bool]:
    """Pack unsigned affine codes into bytes for the supported storage widths."""
    flat = values.flatten().to(torch.uint8)
    if bits == 8:
        return flat.contiguous(), True
    if bits == 4:
        if flat.numel() % 2:
            flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)])
        return (flat[0::2] | (flat[1::2] << 4)).contiguous(), True
    if bits == 2:
        pad = (-flat.numel()) % 4
        if pad:
            flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8, device=flat.device)])
        return (flat[0::4] | (flat[1::4] << 2) | (flat[2::4] << 4) | (flat[3::4] << 6)).contiguous(), True
    raise ValueError(f"cannot pack {bits}-bit values")


def _unpack_unsigned(data: torch.Tensor, bits: int, numel: int) -> torch.Tensor:
    data = data.flatten().to(torch.uint8)
    if bits == 8:
        return data[:numel].to(torch.int16)
    if bits == 4:
        out = torch.empty(data.numel() * 2, dtype=torch.uint8, device=data.device)
        out[0::2] = data & 0x0F
        out[1::2] = (data >> 4) & 0x0F
        return out[:numel].to(torch.int16)
    if bits == 2:
        out = torch.empty(data.numel() * 4, dtype=torch.uint8, device=data.device)
        out[0::4] = data & 0x03
        out[1::4] = (data >> 2) & 0x03
        out[2::4] = (data >> 4) & 0x03
        out[3::4] = (data >> 6) & 0x03
        return out[:numel].to(torch.int16)
    raise ValueError(f"cannot unpack {bits}-bit values")


def _legacy_layout(scale_dim: int) -> StorageLayout:
    """Map the old API's reduction dimension to the KIVI key/value layouts."""
    if scale_dim in {-2, 2}:
        return "keys"
    if scale_dim in {-1, 3}:
        return "values"
    raise ValueError(f"cannot infer KIVI layout from scale_dim={scale_dim}")


@dataclass
class RealQuantizedBlock:
    """A packed asymmetric KIVI-style cache block.

    `minimum` is the affine zero-point offset represented in floating point.
    Keeping it explicitly avoids pretending that an asymmetric format is a
    symmetric format with a shifted integer range.
    """

    data: torch.Tensor
    scale: torch.Tensor
    minimum: torch.Tensor
    shape: tuple[int, ...]
    bits: int
    packed: bool
    numel: int
    dtype: torch.dtype
    layout: StorageLayout | None = None
    group_size: int | None = None

    @property
    def token_count(self) -> int:
        return self.shape[-2]

    def dequantize(self) -> torch.Tensor:
        if self.bits >= 16:
            return self.data.to(self.dtype)
        if self.layout is None or self.group_size is None:
            raise RuntimeError("quantized block is missing KIVI layout metadata")

        codes = _unpack_unsigned(self.data, self.bits, self.numel).reshape(self.shape)
        batch_shape = self.shape[:-2]
        token_count, head_dim = self.shape[-2:]
        if self.layout == "keys":
            grouped = codes.reshape(*batch_shape, token_count // self.group_size, self.group_size, head_dim)
        else:
            grouped = codes.reshape(*batch_shape, token_count, head_dim // self.group_size, self.group_size)
        return (grouped.to(self.scale.dtype) * self.scale + self.minimum).reshape(self.shape).to(self.dtype)

    def storage_breakdown(self) -> dict[str, int]:
        return {
            "payload_bytes": int(self.data.numel() * self.data.element_size()),
            "scale_bytes": int(self.scale.numel() * self.scale.element_size()),
            "minimum_bytes": int(self.minimum.numel() * self.minimum.element_size()),
        }

    def storage_bytes(self) -> int:
        return sum(self.storage_breakdown().values())


def quantize_to_real_block(
    x: torch.Tensor,
    bits: int,
    scale_dim: int | None = None,
    *,
    layout: StorageLayout | None = None,
    group_size: int | None = None,
) -> RealQuantizedBlock:
    """Create a KIVI-compatible affine block from `[B, H, T, D]` cache data.

    Keys are grouped over `group_size` token positions for every channel. Values
    are grouped over `group_size` channels independently for every token.
    """
    _validate_storage_bits(bits)
    if x.ndim != 4:
        raise ValueError(f"expected cache tensor [B, H, T, D], got {tuple(x.shape)}")
    if bits >= 16:
        return RealQuantizedBlock(
            data=x.detach().clone(),
            scale=x.new_empty(0),
            minimum=x.new_empty(0),
            shape=tuple(x.shape),
            bits=16,
            packed=False,
            numel=x.numel(),
            dtype=x.dtype,
        )

    layout = layout or _legacy_layout(scale_dim if scale_dim is not None else -1)
    token_count, head_dim = x.shape[-2:]
    if group_size is None:
        group_size = token_count if layout == "keys" else head_dim
    group_size = int(group_size)
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    grouped_axis = token_count if layout == "keys" else head_dim
    if grouped_axis % group_size:
        raise ValueError(
            f"{layout} grouped dimension {grouped_axis} must be divisible by group_size={group_size}"
        )

    if layout == "keys":
        grouped = x.reshape(*x.shape[:-2], token_count // group_size, group_size, head_dim)
        minimum = grouped.amin(dim=-2, keepdim=True)
        maximum = grouped.amax(dim=-2, keepdim=True)
    else:
        grouped = x.reshape(*x.shape[:-1], head_dim // group_size, group_size)
        minimum = grouped.amin(dim=-1, keepdim=True)
        maximum = grouped.amax(dim=-1, keepdim=True)

    max_code = (1 << bits) - 1
    raw_scale = (maximum - minimum) / max_code
    # A constant group is exactly represented by code 0 plus its minimum.
    scale = torch.where(raw_scale > 0, raw_scale, torch.ones_like(raw_scale))
    codes = torch.round((grouped - minimum) / scale).clamp_(0, max_code).to(torch.uint8)
    packed, is_packed = _pack_unsigned(codes.reshape(x.shape), bits)
    return RealQuantizedBlock(
        data=packed,
        scale=scale.detach(),
        minimum=minimum.detach(),
        shape=tuple(x.shape),
        bits=bits,
        packed=is_packed,
        numel=x.numel(),
        dtype=x.dtype,
        layout=layout,
        group_size=group_size,
    )


@dataclass
class RealQuantizedCacheConfig:
    bits: int
    key_bits: int | None = None
    value_bits: int | None = None
    residual_length: int = 128
    group_size: int | None = None
    # Kept as a compatibility alias for existing notebooks and shell runners.
    block_size: int = 128
    protected_bits: int | None = None
    protected_positions: torch.Tensor | None = None
    protection_target: str = "both"

    @property
    def effective_key_bits(self) -> int:
        return self.bits if self.key_bits is None else self.key_bits

    @property
    def effective_value_bits(self) -> int:
        return self.bits if self.value_bits is None else self.value_bits

    @property
    def effective_group_size(self) -> int:
        return int(self.block_size if self.group_size is None else self.group_size)

    def validate(self, head_dim: int) -> None:
        _validate_storage_bits(self.effective_key_bits)
        _validate_storage_bits(self.effective_value_bits)
        group_size = self.effective_group_size
        if self.residual_length <= 0:
            raise ValueError("KIVI-style real cache requires residual_length > 0")
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.residual_length % group_size:
            raise ValueError("residual_length must be divisible by group_size for key quantization")
        if head_dim % group_size:
            raise ValueError("head dimension must be divisible by group_size for value quantization")


@dataclass
class RealQuantizedCacheLayer(CacheLayerMixin):
    config: RealQuantizedCacheConfig
    key_blocks: list[RealQuantizedBlock] = field(default_factory=list)
    value_blocks: list[RealQuantizedBlock] = field(default_factory=list)
    key_quantized_length: int = 0
    value_quantized_length: int = 0

    is_sliding = False

    def __post_init__(self) -> None:
        CacheLayerMixin.__init__(self)
        self.key_residual: torch.Tensor | None = None
        self.value_residual: torch.Tensor | None = None
        self.dtype: torch.dtype | None = None
        self.device: torch.device | None = None
        self._prefill_complete = False
        self._attention_pending = False

    @property
    def quantized_length(self) -> int:
        """Backward-compatible key-cache prefix length."""
        return self.key_quantized_length

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.config.validate(key_states.shape[-1])
        self.dtype, self.device = key_states.dtype, key_states.device
        self.key_residual = key_states.new_empty((*key_states.shape[:-2], 0, key_states.shape[-1]))
        self.value_residual = value_states.new_empty((*value_states.shape[:-2], 0, value_states.shape[-1]))
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        if not self._prefill_complete:
            self._store_prefill(key_states, value_states)
            # The prefill attention operation must see the original full-precision
            # prompt K/V. Compression only changes the cache retained afterwards.
            return key_states, value_states

        return self._decode_update(key_states, value_states)

    def _store_prefill(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        assert self.key_residual is not None
        assert self.value_residual is not None
        if self.key_quantized_length or self.value_quantized_length or self.key_residual.shape[-2]:
            raise RuntimeError("prefill storage may only be initialized once")

        token_count = key_states.shape[-2]
        residual_length = self.config.residual_length
        # KIVI stores a key prefix in complete residual-sized chunks. The final
        # incomplete chunk stays full precision until it reaches residual_length.
        key_prefix = token_count if token_count % residual_length == 0 else token_count - (token_count % residual_length)
        value_prefix = max(0, token_count - residual_length)
        if key_prefix:
            self._append_key_blocks(key_states[..., :key_prefix, :], start_position=0)
            self.key_quantized_length += key_prefix
        if value_prefix:
            self._append_value_blocks(value_states[..., :value_prefix, :], start_position=0)
            self.value_quantized_length += value_prefix
        self.key_residual = key_states[..., key_prefix:, :].detach().clone()
        self.value_residual = value_states[..., value_prefix:, :].detach().clone()
        self._prefill_complete = True

    def _decode_update(self, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.key_residual is not None
        assert self.value_residual is not None
        if self._attention_pending:
            raise RuntimeError(
                "cache update called before finalize_after_attention(); "
                "the previous decode step has not completed its attention operation"
            )
        self.key_residual = torch.cat([self.key_residual, key_states], dim=-2)
        self.value_residual = torch.cat([self.value_residual, value_states], dim=-2)
        # KIVI evaluates the current attention operation with the newly appended
        # K/V still in the full-precision residual. Storage is sealed explicitly
        # after the model forward has consumed these tensors.
        self._attention_pending = True
        return self.dequantized_keys(), self.dequantized_values()

    def finalize_after_attention(self) -> None:
        """Move eligible residual entries to packed storage after attention."""
        if not self._attention_pending:
            return
        self._seal_key_residual()
        self._seal_value_residual()
        self._attention_pending = False

    def _bits_for_position(self, position: int, default_bits: int, target: str) -> int:
        if self.config.protection_target not in {"both", target}:
            return default_bits
        positions = self.config.protected_positions
        protected_bits = 16 if self.config.protected_bits is None else self.config.protected_bits
        if positions is None or position >= positions.numel() or not bool(positions.flatten()[position].item()):
            return default_bits
        _validate_storage_bits(protected_bits)
        return protected_bits

    def _append_key_blocks(self, tensor: torch.Tensor, start_position: int) -> None:
        group_size = self.config.effective_group_size
        if tensor.shape[-2] % group_size:
            raise ValueError("key block length must be divisible by group_size")
        token_count = tensor.shape[-2]
        run_start = 0
        run_bits: int | None = None
        for group_start in range(0, token_count, group_size):
            group_bits = max(
                self._bits_for_position(start_position + index, self.config.effective_key_bits, "keys")
                for index in range(group_start, group_start + group_size)
            )
            if run_bits is None:
                run_start, run_bits = group_start, group_bits
            elif group_bits != run_bits:
                self.key_blocks.append(
                    quantize_to_real_block(
                        tensor[..., run_start:group_start, :],
                        bits=run_bits,
                        layout="keys",
                        group_size=group_size,
                    )
                )
                run_start, run_bits = group_start, group_bits
        if run_bits is not None:
            self.key_blocks.append(
                quantize_to_real_block(
                    tensor[..., run_start:, :],
                    bits=run_bits,
                    layout="keys",
                    group_size=group_size,
                )
            )

    def _append_value_blocks(self, tensor: torch.Tensor, start_position: int) -> None:
        if tensor.shape[-2] == 0:
            return
        group_size = self.config.effective_group_size
        bits_by_position = [
            self._bits_for_position(start_position + index, self.config.effective_value_bits, "values")
            for index in range(tensor.shape[-2])
        ]
        run_start = 0
        while run_start < tensor.shape[-2]:
            run_bits = bits_by_position[run_start]
            run_end = run_start + 1
            while run_end < tensor.shape[-2] and bits_by_position[run_end] == run_bits:
                run_end += 1
            self.value_blocks.append(
                quantize_to_real_block(
                    tensor[..., run_start:run_end, :],
                    bits=run_bits,
                    layout="values",
                    group_size=group_size,
                )
            )
            run_start = run_end

    def _seal_key_residual(self) -> None:
        assert self.key_residual is not None
        residual_length = self.config.residual_length
        while self.key_residual.shape[-2] >= residual_length:
            block = self.key_residual[..., :residual_length, :]
            self._append_key_blocks(block, start_position=self.key_quantized_length)
            self.key_quantized_length += residual_length
            self.key_residual = self.key_residual[..., residual_length:, :]

    def _seal_value_residual(self) -> None:
        assert self.value_residual is not None
        residual_length = self.config.residual_length
        overflow = self.value_residual.shape[-2] - residual_length
        if overflow <= 0:
            return
        block = self.value_residual[..., :overflow, :]
        self._append_value_blocks(block, start_position=self.value_quantized_length)
        self.value_quantized_length += overflow
        self.value_residual = self.value_residual[..., overflow:, :]

    def dequantized_keys(self) -> torch.Tensor:
        assert self.key_residual is not None
        pieces = [block.dequantize().to(self.key_residual.dtype) for block in self.key_blocks]
        pieces.append(self.key_residual)
        return torch.cat(pieces, dim=-2) if len(pieces) > 1 else self.key_residual

    def dequantized_values(self) -> torch.Tensor:
        assert self.value_residual is not None
        pieces = [block.dequantize().to(self.value_residual.dtype) for block in self.value_blocks]
        pieces.append(self.value_residual)
        return torch.cat(pieces, dim=-2) if len(pieces) > 1 else self.value_residual

    def get_mask_sizes(self, query_length: int, *args: Any, **kwargs: Any) -> tuple[int, int]:
        return self.get_seq_length() + query_length, 0

    def get_seq_length(self) -> int:
        key_residual_length = 0 if self.key_residual is None else int(self.key_residual.shape[-2])
        return self.key_quantized_length + key_residual_length

    def get_max_cache_shape(self) -> int:
        return -1

    def storage_breakdown(self) -> dict[str, int]:
        totals = {
            "key_payload_bytes": 0,
            "key_scale_bytes": 0,
            "key_minimum_bytes": 0,
            "value_payload_bytes": 0,
            "value_scale_bytes": 0,
            "value_minimum_bytes": 0,
            "key_residual_bytes": 0,
            "value_residual_bytes": 0,
        }
        for prefix, blocks in (("key", self.key_blocks), ("value", self.value_blocks)):
            for block in blocks:
                breakdown = block.storage_breakdown()
                totals[f"{prefix}_payload_bytes"] += breakdown["payload_bytes"]
                totals[f"{prefix}_scale_bytes"] += breakdown["scale_bytes"]
                totals[f"{prefix}_minimum_bytes"] += breakdown["minimum_bytes"]
        if self.key_residual is not None:
            totals["key_residual_bytes"] = self.key_residual.numel() * self.key_residual.element_size()
        if self.value_residual is not None:
            totals["value_residual_bytes"] = self.value_residual.numel() * self.value_residual.element_size()
        return totals

    def storage_bytes(self) -> int:
        return sum(self.storage_breakdown().values())


class _RealQuantizedLayerFactory:
    def __init__(self, config: RealQuantizedCacheConfig):
        self.config = config

    def __call__(self) -> RealQuantizedCacheLayer:
        return RealQuantizedCacheLayer(self.config)


class RealQuantizedCache(Cache):
    """KIVI-style persistent storage reference cache.

    This cache stores old K/V in packed affine form but dequantizes blocks before
    standard Hugging Face attention. It supports fidelity and persistent-byte
    experiments; it is deliberately not presented as a fused-kernel speed or
    peak-memory implementation.
    """

    def __init__(self, config: RealQuantizedCacheConfig):
        self.quant_config = config
        super().__init__(layer_class_to_replicate=_RealQuantizedLayerFactory(config))

    def set_protected_positions(self, positions: torch.Tensor | None, protected_bits: int | None = None) -> None:
        self.quant_config.protected_positions = positions
        if protected_bits is not None:
            self.quant_config.protected_bits = protected_bits

    def finalize_after_attention(self) -> None:
        """Finalize every layer after the model forward completes attention."""
        for layer in self.layers:
            if isinstance(layer, RealQuantizedCacheLayer):
                layer.finalize_after_attention()

    def storage_breakdown(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for layer in self.layers:
            if not isinstance(layer, RealQuantizedCacheLayer):
                continue
            for name, value in layer.storage_breakdown().items():
                totals[name] = totals.get(name, 0) + value
        return totals

    def storage_bytes(self) -> int:
        return sum(self.storage_breakdown().values())

    def mean_quantized_prefix_lengths(self) -> dict[str, float]:
        layers = [layer for layer in self.layers if isinstance(layer, RealQuantizedCacheLayer)]
        if not layers:
            return {"keys": 0.0, "values": 0.0}
        return {
            "keys": sum(layer.key_quantized_length for layer in layers) / len(layers),
            "values": sum(layer.value_quantized_length for layer in layers) / len(layers),
        }

    def mean_quantized_prefix_length(self) -> float:
        lengths = self.mean_quantized_prefix_lengths()
        return (lengths["keys"] + lengths["values"]) / 2
