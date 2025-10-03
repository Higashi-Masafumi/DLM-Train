"""
Attention mechanisms for Transformer models

Flash Attention 2を使用した最適化されたアテンション実装。
MHA (Multi-Head Attention), GQA (Grouped-Query Attention), MQA (Multi-Query Attention)をサポート。
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from lightning_utilities.core.imports import RequirementCache

from .config import Config
from .rotary_embedding import apply_rotary_emb_func

RoPECache = tuple[torch.Tensor, torch.Tensor]
KVCache = tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")


class CausalSelfAttention(nn.Module):
    """
    因果的自己アテンション層

    Rotary Position Embeddingを使用し、因果的マスク（未来の情報を見ない）を適用。
    Flash Attention 2による最適化をサポート。

    MHA/GQA/MQAの切り替えはConfigのn_query_groupsで制御:
    - MHA: n_query_groups = n_head（各キー・バリューに対して複数のクエリ）
    - GQA: n_query_groups = n_head // k（グループ化されたクエリ）
    - MQA: n_query_groups = 1（全クエリで共有のキー・バリュー）

    Args:
        config: モデル設定

    Example:
        >>> config = Config(n_head=12, n_embd=768, n_query_groups=12)
        >>> attn = CausalSelfAttention(config)
        >>> x = torch.randn(2, 128, 768)
        >>> rope = (cos, sin)  # Rotary embedding cache
        >>> output, kv_cache = attn(x, rope, max_seq_length=128)
    """

    def __init__(self, config: Config) -> None:
        super().__init__()

        # QKVプロジェクションのサイズ計算
        # Q: n_head個, K: n_query_groups個, V: n_query_groups個
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size

        # QKVプロジェクション（一度にまとめて計算）
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)

        # 出力プロジェクション
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """
        Forward pass

        Args:
            x: 入力テンソル [batch_size, seq_len, n_embd]
            rope: Rotary Embedding cache (cos, sin)
            max_seq_length: 最大シーケンス長
            mask: アテンションマスク（Noneの場合は因果的マスク）
            input_pos: KVキャッシュ使用時の位置インデックス
            kv_cache: KVキャッシュ（推論時）

        Returns:
            output: アテンション出力 [batch_size, seq_len, n_embd]
            kv_cache: 更新されたKVキャッシュ（使用時のみ）
        """
        B, T, C = x.size()  # batch, seq_len, n_embd

        # QKVプロジェクション
        qkv = self.attn(x)

        # MHA/GQA/MQA対応のためのreshape
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

        # KVキャッシュの処理（推論時）
        if kv_cache is not None:
            assert input_pos is not None, "input_pos is required when using kv_cache"
            cache_k, cache_v = kv_cache
            cache_k, cache_v = cache_k.to(dtype=k.dtype), cache_v.to(dtype=v.dtype)

            # トークン数上限チェック
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # キャッシュを1位置左にシフト
                cache_k = torch.roll(cache_k, -1, dims=1)
                cache_v = torch.roll(cache_v, -1, dims=1)

            # キャッシュ更新
            k = cache_k.index_copy_(1, input_pos, k)
            v = cache_v.index_copy_(1, input_pos, v)
            kv_cache = k, v

        # アテンション計算
        y = self.scaled_dot_product_attention(q, k, v, mask=mask)

        # 出力のreshape
        y = y.reshape(B, T, C)

        # 出力プロジェクション
        y = self.proj(y)

        return y, kv_cache

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Scaled Dot-Product Attention

        Flash Attention 2が利用可能な場合は自動的に使用（高速化）

        Args:
            q: Query [B, T, n_head, head_size]
            k: Key [B, T, n_query_groups, head_size]
            v: Value [B, T, n_query_groups, head_size]
            mask: アテンションマスク

        Returns:
            アテンション出力 [B, T, n_head, head_size]
        """
        scale = 1.0 / math.sqrt(self.config.head_size)

        # Flash Attention 2の使用判定
        use_flash = (
            FlashAttention2Available
            and mask is None
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        )

        if use_flash:
            from flash_attn import flash_attn_func
            return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True)

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
            attn_mask=mask,
            dropout_p=0.0,
            scale=scale,
            is_causal=(mask is None)
        )

        return y.transpose(1, 2)  # [B, T, n_head, head_size]
