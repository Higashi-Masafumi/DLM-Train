"""
Transformer Block implementation

Diffusion LMのTransformerブロック実装。
"""

import torch
import torch.nn as nn

from .config import Config

RoPECache = tuple[torch.Tensor, torch.Tensor]


class Block(nn.Module):
    """
    Transformer Block

    Attention + MLPの標準的なTransformerブロック。
    Parallel ResidualとSequential Residualの両方をサポート。

    Args:
        config: モデル設定

    Example:
        >>> config = Config.from_name("Diff_LLaMA_170M")
        >>> block = Block(config)
        >>> x = torch.randn(2, 128, 768)
        >>> rope = (cos, sin)
        >>> output = block(x, rope)
    """

    def __init__(self, config: Config) -> None:
        super().__init__()

        # 最初の正規化層（Attention前）
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)

        # Self Attention
        # diffusionlm.pyのSelfAttentionを使用（causal=False）
        from .diffusion_attention import SelfAttention
        self.attn = SelfAttention(config)

        # 2つ目の正規化層（MLP前）
        # shared_attention_normの場合は共有
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)

        # MLP
        self.mlp = config.mlp_class(config)

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
            出力テンソル [batch_size, seq_len, n_embd]
        """
        # Pre-normalization
        n_1 = self.norm_1(x)

        # Self Attention
        h = self.attn(n_1, rope)

        if self.config.parallel_residual:
            # Parallel Residual (GPT-NeoX style)
            # x_out = x + Attention(norm(x)) + MLP(norm(x))
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            x = x + h + self.mlp(n_2)
        else:
            # Sequential Residual (標準的なTransformer)
            # x_out = x + MLP(norm(x + Attention(norm(x))))
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm)."
                )

            x = x + h
            x = x + self.mlp(self.norm_2(x))

        return x


__all__ = ["Block"]
