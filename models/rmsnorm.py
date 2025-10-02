import torch
from torch.nn import init

# FlashAttention v2 公開 API
from flash_attn.ops.layer_norm import (
    dropout_add_layer_norm as fa_dropout_add_layer_norm,
    dropout_add_layer_norm_parallel_residual as fa_dropout_add_layer_norm_parallel_residual,
)
from flash_attn.ops.rms_norm import rms_norm as fa_rms_norm

# ─────────────────────────────────────────────────────────────────────────────
# 便利ラッパ（v1 風の関数名を保ちつつ、中身は v2 の公開 API を呼ぶ）
# ─────────────────────────────────────────────────────────────────────────────

def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    epsilon: float
) -> torch.Tensor:
    """residual=None, dropout=0.0 の純 LayerNorm 相当"""
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
    layerscale: torch.Tensor | None = None,  # (= colscale)
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    return_dropout_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """v1 互換の関数名で v2 API を薄くラップ"""
    return fa_dropout_add_layer_norm(
        x0,
        residual,
        weight,
        bias,
        dropout_p=dropout_p,
        epsilon=epsilon,
        rowscale=rowscale,
        layerscale=layerscale,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
        return_dropout_mask=return_dropout_mask,
    )


def dropout_add_layer_norm_parallel_residual(
    x0: torch.Tensor,
    x1: torch.Tensor | None,
    residual: torch.Tensor | None,
    weight0: torch.Tensor,
    bias0: torch.Tensor | None,
    weight1: torch.Tensor | None,
    bias1: torch.Tensor | None,
    dropout_p: float,
    epsilon: float,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    return_dropout_mask: bool = False,
):
    """parallel residual 版：v1 関数名で v2 API を呼ぶ"""
    return fa_dropout_add_layer_norm_parallel_residual(
        x0,
        x1,
        residual,
        weight0,
        bias0,
        weight1,
        bias1,
        dropout_p=dropout_p,
        epsilon=epsilon,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
        return_dropout_mask=return_dropout_mask,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Subset 版（v2 には専用 fused が無いので安全な Python 実装で代替）
# 抽出 → fused 呼び出し → scatter。速度は落ちますが挙動は揃います。
# ─────────────────────────────────────────────────────────────────────────────

def dropout_add_layer_norm_subset(
    x0: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dropout_p: float,
    epsilon: float,
    layerscale: torch.Tensor | None = None,      # (= colscale)
    x0_subset: torch.Tensor | None = None,       # 1D indices
    out_subset: torch.Tensor | None = None,      # 1D indices
    rowscale_const: float = 1.0,                 # 互換ダミー（未使用）
    out_numrows: int = 0,                        # 互換ダミー（未使用）
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    return_dropout_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if x0_subset is None and out_subset is None and layerscale is None:
        return dropout_add_layer_norm(
            x0, residual, weight, bias, dropout_p, epsilon,
            rowscale=None, layerscale=None, prenorm=prenorm,
            residual_in_fp32=residual_in_fp32, return_dropout_mask=return_dropout_mask
        )

    if x0_subset is None or out_subset is None:
        raise ValueError("subset を使う場合は x0_subset と out_subset の両方が必要です。")

    H = weight.numel()
    x0m = x0.view(-1, H)
    residualm = residual.view(-1, H) if residual is not None else None

    x0_idx = x0_subset.to(torch.long).to(x0m.device)
    out_idx = out_subset.to(torch.long).to(x0m.device)

    x0_sel = x0m.index_select(0, x0_idx)
    residual_sel = residualm.index_select(0, x0_idx) if residualm is not None else None

    if layerscale is not None:
        x0_sel = x0_sel * layerscale

    z_sel = fa_dropout_add_layer_norm(
        x0_sel,
        residual_sel,
        weight,
        bias,
        dropout_p=dropout_p,
        epsilon=epsilon,
        rowscale=None,
        layerscale=None,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
        return_dropout_mask=return_dropout_mask,
    )

    outm = x0m.new_empty(x0m.shape)
    if return_dropout_mask:
        z_val, dmask_sel = z_sel
        outm.index_copy_(0, out_idx, z_val)
        out = outm.view(x0.shape)
        dmask_full = x0.new_zeros(x0.shape, dtype=torch.uint8)
        dmask_full.view(-1, H).index_copy_(0, out_idx, dmask_sel.to(torch.uint8))
        return out, dmask_full
    else:
        outm.index_copy_(0, out_idx, z_sel)
        return outm.view(x0.shape)

# ─────────────────────────────────────────────────────────────────────────────
# nn.Module ラッパ
# ─────────────────────────────────────────────────────────────────────────────

class DropoutAddLayerNorm(torch.nn.Module):
    """v1 のクラス名/使い勝手のまま v2 の fused を呼ぶ"""
    def __init__(
        self,
        hidden_size: int,
        prenorm: bool = False,
        p: float = 0.0,
        eps: float = 1e-5,
        residual_in_fp32: bool = False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.prenorm = prenorm
        self.p = p
        self.eps = eps
        self.residual_in_fp32 = residual_in_fp32
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x0: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        return dropout_add_layer_norm(
            x0,
            residual,
            self.weight,
            self.bias,
            self.p if self.training else 0.0,
            self.eps,
            prenorm=self.prenorm,
            residual_in_fp32=self.residual_in_fp32,
        )

# ─────────────────────────────────────────────────────────────────────────────
# RMSNorm（fused 版）
# ─────────────────────────────────────────────────────────────────────────────

def rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    return fa_rms_norm(x, weight, epsilon)

class FusedRMSNorm(torch.nn.Module):
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.dim = dim
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


class RMSNorm(torch.nn.Module):
    """fused を使わない参照実装（互換）"""
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
