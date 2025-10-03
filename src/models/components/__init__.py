"""
Model Components Package

Transformerモデルの基本コンポーネントを提供します。

主要モジュール:
- config: モデル設定（Pydantic）
- attention: CausalSelfAttention（Flash Attention 2対応）
- diffusion_attention: SelfAttention（Non-causal, Diffusion LM用）
- mlp: GptNeoxMLP, LLaMAMLP
- rmsnorm: RMSNorm, LayerNorm
- rotary_embedding: Rotary Position Embedding
- block: Transformer Block
- encoder: TransEncoder
"""

from .attention import CausalSelfAttention, KVCache, RoPECache
from .block import Block
from .config import Config, configs, name_to_config
from .diffusion_attention import SelfAttention
from .encoder import TransEncoder
from .mlp import GptNeoxMLP, GptNeoXMLP, LLaMAMLP
from .rmsnorm import FusedRMSNorm, LayerNorm, RMSNorm, layer_norm, rms_norm
from .rotary_embedding import (
    ApplyRotaryEmb,
    RotaryEmbedding,
    apply_rotary_emb_func,
    build_rope_cache,
)

__all__ = [
    # Config
    "Config",
    "configs",
    "name_to_config",
    # Attention
    "CausalSelfAttention",
    "SelfAttention",
    "RoPECache",
    "KVCache",
    # MLP
    "GptNeoxMLP",
    "GptNeoXMLP",
    "LLaMAMLP",
    # Normalization
    "RMSNorm",
    "FusedRMSNorm",
    "LayerNorm",
    "rms_norm",
    "layer_norm",
    # Rotary Embedding
    "ApplyRotaryEmb",
    "RotaryEmbedding",
    "apply_rotary_emb_func",
    "build_rope_cache",
    # Block & Encoder
    "Block",
    "TransEncoder",
]

from .attention import CausalSelfAttention, KVCache, RoPECache
from .config import Config, configs, name_to_config
from .mlp import GptNeoxMLP, GptNeoXMLP, LLaMAMLP
from .rmsnorm import FusedRMSNorm, LayerNorm, RMSNorm, layer_norm, rms_norm
from .rotary_embedding import (
    ApplyRotaryEmb,
    RotaryEmbedding,
    apply_rotary_emb_func,
    build_rope_cache,
)

__all__ = [
    # Config
    "Config",
    "configs",
    "name_to_config",
    # Attention
    "CausalSelfAttention",
    "RoPECache",
    "KVCache",
    # MLP
    "GptNeoxMLP",
    "GptNeoXMLP",
    "LLaMAMLP",
    # Normalization
    "RMSNorm",
    "FusedRMSNorm",
    "LayerNorm",
    "rms_norm",
    "layer_norm",
    # Rotary Embedding
    "ApplyRotaryEmb",
    "RotaryEmbedding",
    "apply_rotary_emb_func",
    "build_rope_cache",
]
