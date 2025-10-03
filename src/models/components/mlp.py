"""
MLP (Multi-Layer Perceptron) modules

Transformer内のFeed-Forwardネットワークの実装。
GPT-NeoXスタイルとLLaMAスタイル（SwiGLU）をサポート。
"""

import torch
import torch.nn as nn
from xformers.ops import SwiGLU

from .config import Config


class GptNeoxMLP(nn.Module):
    """
    GPT-NeoX style MLP

    標準的な2層FFN with GELU activation:
    x -> Linear -> GELU -> Linear -> output

    Args:
        config: モデル設定

    Example:
        >>> config = Config(n_embd=768, intermediate_size=3072)
        >>> mlp = GptNeoxMLP(config)
        >>> x = torch.randn(2, 128, 768)
        >>> output = mlp(x)
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.intermediate_size is not None, "intermediate_size must be set for GptNeoxMLP"

        # 第1層: n_embd -> intermediate_size
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)

        # 第2層: intermediate_size -> n_embd
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: 入力テンソル [batch_size, seq_len, n_embd]

        Returns:
            出力テンソル [batch_size, seq_len, n_embd]
        """
        x = self.fc(x)
        x = torch.nn.functional.gelu(x)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    """
    LLaMA style MLP with SwiGLU activation

    SwiGLU (Swish-Gated Linear Unit)を使用した高性能なFFN。
    LLaMA, PaLM等の最新モデルで使用されている活性化関数。

    SwiGLU: x -> (W1(x) * silu(W2(x))) -> W3 -> output

    Args:
        config: モデル設定

    Note:
        xformersライブラリのSwiGLU実装を使用（最適化済み）

    Example:
        >>> config = Config(n_embd=768, intermediate_size=2048, mlp_class_name="LLaMAMLP")
        >>> mlp = LLaMAMLP(config)
        >>> x = torch.randn(2, 128, 768)
        >>> output = mlp(x)
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.intermediate_size is not None, "intermediate_size must be set for LLaMAMLP"

        # SwiGLU activation
        # xformers実装を使用（融合カーネルで高速化）
        self.swiglu = SwiGLU(
            config.n_embd,
            config.intermediate_size,
            bias=False,  # LLaMAではバイアスなし
            _pack_weights=False,  # 重みを個別に保持
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: 入力テンソル [batch_size, seq_len, n_embd]

        Returns:
            出力テンソル [batch_size, seq_len, n_embd]
        """
        return self.swiglu(x)


# エイリアス（後方互換性のため）
GptNeoXMLP = GptNeoxMLP

__all__ = ["GptNeoxMLP", "GptNeoXMLP", "LLaMAMLP"]
