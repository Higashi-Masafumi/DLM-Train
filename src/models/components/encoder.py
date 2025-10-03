"""
Diffusion Language Model Encoder

TransEncoderの実装（Diffusion LM用）。
"""

import math

import torch
import torch.nn as nn
from xformers.ops import SwiGLU

from .block import Block
from .config import Config
from .diffusion_attention import SelfAttention
from .mlp import LLaMAMLP
from .rotary_embedding import build_rope_cache

RoPECache = tuple[torch.Tensor, torch.Tensor]


class TransEncoder(nn.Module):
    """
    Transformer Encoder for Diffusion Language Model

    Diffusion Language Model用のTransformerエンコーダー。
    埋め込み層、複数のTransformerブロック、最終正規化層、LMヘッドから構成。

    Args:
        config: モデル設定

    Example:
        >>> config = Config.from_name("Diff_LLaMA_170M")
        >>> model = TransEncoder(config)
        >>> idx = torch.randint(0, 32000, (2, 128))
        >>> logits = model(idx)
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        # LM Head (出力層)
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)

        # Transformer layers
        self.transformer = nn.ModuleDict(
            dict(
                # Token Embedding (+1 for mask token)
                wte=nn.Embedding(config.padded_vocab_size + 1, config.n_embd),
                # Transformer blocks
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                # Final normalization
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )

        # Rotary Embedding cache
        self.rope_cache: RoPECache | None = None

    def _init_weights(self, module: nn.Module, n_layer: int) -> None:
        """
        重みの初期化（GPT-NeoXスタイル）

        Args:
            module: 初期化対象のモジュール
            n_layer: レイヤー数（深さに応じたスケーリング用）
        """
        # Embedding層の初期化
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=math.sqrt(2.0 / 5 / self.config.n_embd)
            )

        # Linear層の初期化
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=math.sqrt(2.0 / 5 / self.config.n_embd)
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        # 出力層の特別な初期化（深さに応じてスケール）
        for name, p in module.named_parameters():
            # MLPの出力層またはAttentionの出力層
            is_output_layer = (
                (name == "proj.weight" and isinstance(module, LLaMAMLP)) or
                (name == "w3.weight" and isinstance(module, SwiGLU)) or
                (name == "proj.weight" and isinstance(module, SelfAttention))
            )

            if is_output_layer:
                nn.init.normal_(
                    p,
                    mean=0.0,
                    std=1 / math.sqrt(self.config.n_embd) / n_layer
                )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            idx: 入力トークンID [batch_size, seq_len]

        Returns:
            logits: 出力ロジット [batch_size, seq_len, vocab_size]
        """
        B, T = idx.size()

        block_size = self.config.block_size
        assert T <= block_size, \
            f"Cannot forward sequence of length {T}, block size is only {block_size}"

        # Rotary Embedding cacheの構築（初回のみ）
        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)

        # Rotary Embedding
        cos, sin = self.rope_cache
        cos = cos[:T]
        sin = sin[:T]
        rope = (cos, sin)

        # Token Embedding
        x = self.transformer.wte(idx)  # [B, T, n_embd]

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x, rope)

        # Final normalization
        x = self.transformer.ln_f(x)

        # LM head
        logits = self.lm_head(x)  # [B, T, vocab_size]

        return logits

    def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
        """
        Rotary Embedding cacheの構築

        Args:
            idx: 入力テンソル（デバイス情報取得用）

        Returns:
            (cos, sin)のタプル
        """
        return build_rope_cache(
            seq_len=self.config.block_size,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=torch.bfloat16,
            device=idx.device,
            base=10000,
            condense_ratio=self.config.condense_ratio,
        )


__all__ = ["TransEncoder"]
