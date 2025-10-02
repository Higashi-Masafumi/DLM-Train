import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self

from models.config import Config


class SwiGLU(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.up_proj = nn.Linear(in_features, out_features * 2, bias=True)
        self.down_proj = nn.Linear(out_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_u = self.up_proj(x)
        v, g = x_u.chunk(2, dim=-1)
        y = F.silu(g) * v
        return self.down_proj(y)
