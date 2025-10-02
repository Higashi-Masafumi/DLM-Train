from typing import Any

import torch
from flash_attn.layers.rotary import apply_rotary_emb  # ここからimportすればOK


class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, interleaved=False, inplace=False):
        """
        x: (batch_size, seqlen, nheads, headdim)
        cos, sin: (seqlen, rotary_dim / 2)
        interleaved: GPT-J styleかGPT-NeoX styleか
        rotary_dim must be <= headdim
        """
        seqlen = x.shape[1]

        out = apply_rotary_emb(
            x,
            cos[:seqlen],  # (seqlen, rotary_dim/2)
            sin[:seqlen],  # (seqlen, rotary_dim/2)
            interleaved=interleaved,
            inplace=inplace,
        )

        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return grad_outputs[0], None, None, None, None


# これを呼び出し元で使えばOK
apply_rotary_emb_func = ApplyRotaryEmb.apply
