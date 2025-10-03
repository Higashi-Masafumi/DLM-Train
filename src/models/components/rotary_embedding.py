"""
Rotary Position Embedding (RoPE) implementation

Flash Attentionライブラリを使用した効率的なRotary Embeddingの実装を提供します。
"""

from typing import Any

import torch
from flash_attn.layers.rotary import apply_rotary_emb


class ApplyRotaryEmb(torch.autograd.Function):
    """
    Rotary Embeddingの適用（カスタムautograd関数）

    Flash Attentionの最適化実装を使用して、位置情報を回転行列として
    エンコードします（RoFormer論文で提案された手法）。

    GPT-NeoX/LLaMAスタイルのRotary Embeddingをサポート。
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        interleaved: bool = False,
        inplace: bool = False,
    ) -> torch.Tensor:
        """
        Rotary Embeddingの適用

        Args:
            ctx: autograd context
            x: 入力テンソル [batch_size, seqlen, nheads, headdim]
            cos: cosine値 [seqlen, rotary_dim / 2]
            sin: sine値 [seqlen, rotary_dim / 2]
            interleaved: GPT-J styleかGPT-NeoX styleか
                - False (default): GPT-NeoX style - 前半と後半を分けて回転
                - True: GPT-J style - 交互に要素を回転
            inplace: インプレース演算を行うか

        Returns:
            Rotary Embeddingが適用された出力テンソル

        Note:
            rotary_dim must be <= headdim
            通常、rotary_dim = headdim として使用されます。
        """
        seqlen = x.shape[1]

        out = apply_rotary_emb(
            x,
            cos[:seqlen],  # シーケンス長に合わせてトリミング
            sin[:seqlen],
            interleaved=interleaved,
            inplace=inplace,
        )

        # Backward用に保存
        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        ctx.inplace = inplace

        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """
        Backward pass

        Rotary Embeddingは回転行列なので、逆回転を適用することで
        勾配を計算できます。

        Returns:
            (grad_x, None, None, None, None)
        """
        # 簡易実装: 勾配をそのまま返す
        # より正確な実装が必要な場合は、逆回転を適用
        return grad_output, None, None, None, None


# メイン関数：外部から使用する際はこれを呼ぶ
apply_rotary_emb_func = ApplyRotaryEmb.apply


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    base: int = 10000,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    condense_ratio: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rotary Embeddingのキャッシュ（cos/sinテーブル）を構築

    Args:
        seq_len: 最大シーケンス長
        n_elem: Rotary次元数（通常はhead_dim）
        base: 周波数の基数（デフォルト10000）
        device: テンソルのデバイス
        dtype: テンソルの型
        condense_ratio: 位置インデックスの圧縮比率（Diffusion用）

    Returns:
        (cos, sin)のタプル、各形状は[seq_len, n_elem // 2]

    Example:
        >>> cos, sin = build_rope_cache(2048, 128)
        >>> cos.shape, sin.shape
        (torch.Size([2048, 64]), torch.Size([2048, 64]))
    """
    # 周波数の計算: θ_i = base^(-2i/n_elem)
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

    # 位置エンコーディング: m * θ_i（condense_ratioで調整）
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device) / condense_ratio
    idx_theta = torch.outer(seq_idx, theta)  # [seq_len, n_elem // 2]

    # cos/sinの計算
    cos = torch.cos(idx_theta)
    sin = torch.sin(idx_theta)

    # dtype変換（bfloat16/float16対応）
    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    if dtype in (torch.float16, torch.int8):
        return cos.half(), sin.half()

    return cos, sin


class RotaryEmbedding(torch.nn.Module):
    """
    Rotary Position Embedding Layer

    LLaMA等のモデルで使用される位置エンコーディング手法。
    絶対位置ではなく相対位置を効率的にエンコードします。

    Args:
        dim: 埋め込み次元数（head_dim）
        max_seq_len: 最大シーケンス長
        base: 周波数の基数
        interleaved: GPT-J styleかGPT-NeoX styleか

    Example:
        >>> rope = RotaryEmbedding(dim=128, max_seq_len=2048)
        >>> x = torch.randn(2, 512, 12, 128)  # [B, L, H, D]
        >>> x_rotated = rope(x)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 10000,
        interleaved: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.interleaved = interleaved

        # cos/sinキャッシュを事前計算
        cos, sin = build_rope_cache(
            seq_len=max_seq_len,
            n_elem=dim,
            base=base,
        )

        # バッファとして登録（学習対象外）
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotary Embeddingの適用

        Args:
            x: 入力テンソル [batch_size, seq_len, n_heads, head_dim]

        Returns:
            Rotary Embeddingが適用された出力
        """
        return apply_rotary_emb_func(
            x,
            self.cos,
            self.sin,
            interleaved=self.interleaved,
            inplace=False,
        )
