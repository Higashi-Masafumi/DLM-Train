"""
RMSNorm (Root Mean Square Layer Normalization) implementations

Flash Attention v2 APIを使用した最適化されたRMSNorm実装を提供します。
"""

import torch
import torch.nn as nn

# FlashAttention v2 公開 API
from flash_attn.ops.layer_norm import (
    dropout_add_layer_norm as fa_dropout_add_layer_norm,
)
from flash_attn.ops.layer_norm import (
    dropout_add_layer_norm_parallel_residual as fa_dropout_add_layer_norm_parallel_residual,
)
from flash_attn.ops.rms_norm import rms_norm as fa_rms_norm
from torch.nn import init

# ─────────────────────────────────────────────────────────────────────────────
# 便利ラッパ（v1 風の関数名を保ちつつ、中身は v2 の公開 API を呼ぶ）
# ─────────────────────────────────────────────────────────────────────────────


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    epsilon: float
) -> torch.Tensor:
    """
    純粋なLayerNorm（residual=None, dropout=0.0）

    Args:
        x: 入力テンソル
        weight: スケールパラメータ
        bias: バイアスパラメータ（Noneも可）
        epsilon: 数値安定性のための小さな値

    Returns:
        正規化された出力
    """
    return fa_dropout_add_layer_norm(
        x,                      # x0
        None,                   # residual
        weight,
        bias,
        dropout_p=0.0,
        epsilon=epsilon,
        rowscale=None,
        layerscale=None,
        prenorm=False,
        residual_in_fp32=False,
        return_dropout_mask=False,
    )


def dropout_add_layer_norm(
    x0: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dropout_p: float,
    epsilon: float,
    rowscale: torch.Tensor | None = None,
    layerscale: torch.Tensor | None = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    return_dropout_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Dropout + Residual + LayerNormの融合演算

    v1互換の関数名でv2 APIをラップ
    """
    return fa_dropout_add_layer_norm(
        x0,
        residual,
        weight,
        bias,
        dropout_p,
        epsilon,
        rowscale,
        layerscale,
        prenorm,
        residual_in_fp32,
        return_dropout_mask,
    )


def dropout_add_layer_norm_parallel_residual(
    x0: torch.Tensor,
    x1: torch.Tensor,
    residual: torch.Tensor | None,
    weight0: torch.Tensor,
    bias0: torch.Tensor | None,
    weight1: torch.Tensor,
    bias1: torch.Tensor | None,
    dropout_p: float,
    epsilon: float,
    rowscale: torch.Tensor | None = None,
    layerscale0: torch.Tensor | None = None,
    layerscale1: torch.Tensor | None = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    return_dropout_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Parallel Residual構造用のLayerNorm

    GPT-NeoX等で使用される、2つの並列パスを持つアーキテクチャ向け
    """
    return fa_dropout_add_layer_norm_parallel_residual(
        x0,
        x1,
        residual,
        weight0,
        bias0,
        weight1,
        bias1,
        dropout_p,
        epsilon,
        rowscale,
        layerscale0,
        layerscale1,
        prenorm,
        residual_in_fp32,
        return_dropout_mask,
    )


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    epsilon: float,
) -> torch.Tensor:
    """
    RMSNorm実装（Flash Attention v2最適化版）

    LayerNormの簡略版で、平均中心化を行わずRMSのみで正規化
    LLaMA等のモデルで使用

    Args:
        x: 入力テンソル [..., hidden_size]
        weight: スケールパラメータ [hidden_size]
        bias: バイアスパラメータ（通常はNone）
        epsilon: 数値安定性のための小さな値

    Returns:
        正規化された出力
    """
    # Flash Attentionのrms_normはbiasをサポートしていないため、
    # biasが必要な場合は別途追加
    output = fa_rms_norm(x, weight, epsilon)
    if bias is not None:
        output = output + bias
    return output


# ─────────────────────────────────────────────────────────────────────────────
# nn.Module ラッパー
# ─────────────────────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    """
    RMSNorm Layer（PyTorchモジュール版）

    Flash Attention最適化版のRMSNormを使用

    Args:
        normalized_shape: 正規化する次元のサイズ（通常はhidden_size）
        eps: 数値安定性のための小さな値
        bias: バイアスを使用するか（通常はFalse）

    Example:
        >>> norm = RMSNorm(768)
        >>> x = torch.randn(2, 128, 768)
        >>> output = norm(x)
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        """パラメータの初期化"""
        init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: 入力テンソル [..., normalized_shape]

        Returns:
            正規化された出力
        """
        return rms_norm(x, self.weight, self.bias, self.eps)


class FusedRMSNorm(RMSNorm):
    """
    Fused RMSNorm（RMSNormと同じ）

    後方互換性のために残しているエイリアス。
    実装はRMSNormと同じ（Flash Attention最適化版を使用）
    """
    pass


class LayerNorm(nn.Module):
    """
    LayerNorm（Flash Attention最適化版）

    標準のnn.LayerNormより高速な実装

    Args:
        normalized_shape: 正規化する次元のサイズ
        eps: 数値安定性のための小さな値
        bias: バイアスを使用するか
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        """パラメータの初期化"""
        init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: 入力テンソル [..., normalized_shape]

        Returns:
            正規化された出力
        """
        return layer_norm(x, self.weight, self.bias, self.eps)
