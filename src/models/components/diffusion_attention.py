"""
Diffusion LM Attention (non-causal)

Diffusion Language Model用のnon-causal attention実装。
通常のTransformerと異なり、因果的マスクを使用しない。
"""

import math

import torch
import torch.nn as nn
from flash_attn import flash_attn_func
from lightning_utilities.core.imports import RequirementCache

from .config import Config
from .rotary_embedding import apply_rotary_emb_func

RoPECache = tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")


class SelfAttention(nn.Module):
    """
    Non-causal Self Attention (Diffusion LM用)

    Diffusion Language Modelで使用される、因果的マスクを使わないアテンション。
    通常のCausalSelfAttentionと異なり、全トークンを参照可能。

    Args:
        config: モデル設定

    Note:
        CausalSelfAttentionとの違い:
        - causal=Falseでアテンションを計算
        - is_causal=Falseを使用
        - 未来のトークンも参照可能

    Example:
        >>> config = Config.from_name("Diff_LLaMA_170M")
        >>> attn = SelfAttention(config)
        >>> x = torch.randn(2, 128, 768)
        >>> rope = (cos, sin)
        >>> output = attn(x, rope)
    """

    def __init__(self, config: Config) -> None:
        super().__init__()

        # QKVプロジェクションのサイズ計算
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size

        # QKVプロジェクション
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)

        # 出力プロジェクション
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: 入力テンソル [batch_size, seq_len, n_embd]
            rope: Rotary Embedding cache (cos, sin)

        Returns:
            アテンション出力 [batch_size, seq_len, n_embd]
        """
        B, T, C = x.size()  # batch, seq_len, n_embd

        # QKVプロジェクション
        qkv = self.attn(x)

        # MHA/GQA/MQA対応のreshape
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # Q + K + V
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)

        # QKVに分割
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)

        # Flash Attention用にreshape
        q = q.reshape(B, T, -1, self.config.head_size)  # [B, T, n_head, head_size]
        k = k.reshape(B, T, -1, self.config.head_size)  # [B, T, n_query_groups, head_size]
        v = v.reshape(B, T, -1, self.config.head_size)

        # Rotary Embeddingの適用
        cos, sin = rope
        q = apply_rotary_emb_func(q, cos, sin, False, True)
        k = apply_rotary_emb_func(k, cos, sin, False, True)

        # アテンション計算（non-causal）
        y = self.scaled_dot_product_attention(q, k, v)

        # 出力のreshape
        y = y.reshape(B, T, C)

        # 出力プロジェクション
        y = self.proj(y)

        return y

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Scaled Dot-Product Attention (Non-causal)

        Flash Attention 2が利用可能な場合は自動的に使用。
        causal=Falseで全トークンを参照可能。

        Args:
            q: Query [B, T, n_head, head_size]
            k: Key [B, T, n_query_groups, head_size]
            v: Value [B, T, n_query_groups, head_size]

        Returns:
            アテンション出力 [B, T, n_head, head_size]
        """
        scale = 1.0 / math.sqrt(self.config.head_size)

        # Flash Attention 2の使用判定
        use_flash = (
            FlashAttention2Available
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        )

        if use_flash:
            # Non-causal attention (causal=False)
            return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=False)

        # 標準のScaled Dot-Product Attention
        q = q.transpose(1, 2)  # [B, n_head, T, head_size]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # GQA/MQA対応: KeyとValueを繰り返し
        if q.size() != k.size():
            repeat_factor = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            scale=scale,
            is_causal=False  # Non-causal attention
        )

        return y.transpose(1, 2)  # [B, T, n_head, head_size]


__all__ = ["SelfAttention"]
