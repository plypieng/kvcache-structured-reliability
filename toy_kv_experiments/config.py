from dataclasses import dataclass


@dataclass
class TinyGPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    dropout: float = 0.0
