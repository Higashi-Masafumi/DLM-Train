"""Learning rate schedulers for training.

このモジュールは、トレーニングで使用される学習率スケジューラーを提供します。
"""

import math
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer


class CosineAnnealingWithWarmup(LambdaLR):
    """Cosine annealing学習率スケジューラーとウォームアップ付き.
    
    ウォームアップ期間中は線形に学習率を増加させ、その後コサインアニーリングで減衰させます。
    
    Args:
        optimizer: PyTorchオプティマイザー
        warmup_steps: ウォームアップステップ数
        max_steps: 最大トレーニングステップ数
        min_lr: 最小学習率（ベース学習率に対する比率）
        last_epoch: 最後のエポック番号（resumeする場合）
    
    Note:
        この実装はPyTorch LightningのLRSchedulerConfigと互換性があります。
        interval='step', frequency=1 で使用することを想定しています。
    
    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        >>> scheduler = CosineAnnealingWithWarmup(
        ...     optimizer, warmup_steps=100, max_steps=1000, min_lr=0.1
        ... )
        >>> for step in range(max_steps):
        ...     loss.backward()
        ...     optimizer.step()
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        
        super().__init__(optimizer, self._lr_lambda, last_epoch)

    def _lr_lambda(self, current_step: int) -> float:
        """学習率の乗数を計算.
        
        Args:
            current_step: 現在のステップ数
            
        Returns:
            学習率の乗数（ベース学習率に対する比率）
        """
        # ウォームアップ期間中は線形に増加
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        
        # ウォームアップ後はコサインアニーリング
        progress = float(current_step - self.warmup_steps) / float(
            max(1, self.max_steps - self.warmup_steps)
        )
        progress = min(progress, 1.0)  # max_stepsを超えても1.0以下にクリップ
        
        # コサイン曲線: 1.0 -> min_lr
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (1.0 - self.min_lr) * cosine_decay
