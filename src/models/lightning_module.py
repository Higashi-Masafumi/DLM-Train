"""
PyTorch Lightning Module for Diffusion Language Model

このモジュールは既存のTransEncoderをPyTorch LightningModuleでラップし、
学習・検証ループ、オプティマイザ設定、メトリクスログ記録を統合します。
"""

from typing import Any

import lightning as L
import torch
import torch.nn as nn
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .components.config import Config
from .components.encoder import TransEncoder


class DiffusionLMLightningModule(L.LightningModule):
    """
    Diffusion Language Model用のPyTorch Lightning Module

    既存のTransEncoderモデルをラップし、Lightning形式のトレーニングを提供します。

    Args:
        model_config: モデルアーキテクチャの設定（Config互換の辞書）
        optimizer_config: オプティマイザの設定（lr, weight_decay等）
        scheduler_config: スケジューラの設定（warmup_steps, max_steps等）
        num_parameters: モデルパラメータ数（FLOPs計算用）
    """

    def __init__(
        self,
        model_config: dict[str, Any],
        optimizer_config: dict[str, Any] | None = None,
        scheduler_config: dict[str, Any] | None = None,
        num_parameters: float | None = None,
    ):
        super().__init__()

        # Hydraから渡された設定を保存
        self.save_hyperparameters()

        # Config オブジェクトに変換
        # TODO: Pydanticモデルとして適切に変換
        self.model_config = self._convert_to_config(model_config)

        # TransEncoderモデルの初期化
        self.model = TransEncoder(self.model_config)

        # Loss関数（Flash Attention最適化版）
        self.criterion = CrossEntropyLoss()

        # パラメータ数を保存（ログ用）
        self.num_parameters = num_parameters or self._count_parameters()

        # オプティマイザ・スケジューラ設定
        self.optimizer_config = optimizer_config or {}
        self.scheduler_config = scheduler_config or {}

    def _convert_to_config(self, config_dict: dict[str, Any]) -> Config:
        """
        辞書形式の設定をConfigオブジェクトに変換

        TODO: より堅牢な変換ロジックを実装
        """
        # 簡易実装: 既存のConfigクラスを使用
        # 実際にはPydanticのvalidationを活用
        return Config(**config_dict)

    def _count_parameters(self) -> int:
        """モデルのパラメータ数を計算"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        順伝播

        Args:
            idx: 入力トークンID [batch_size, seq_len]

        Returns:
            logits: 出力ロジット [batch_size, seq_len, vocab_size]
        """
        return self.model(idx)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        トレーニングステップ

        Args:
            batch: (input_ids, targets)のタプル
            batch_idx: バッチインデックス

        Returns:
            loss: 計算された損失
        """
        input_ids, targets = batch

        # 順伝播
        logits = self(input_ids)

        # Loss計算（Flash Attention最適化版）
        # logits: [batch_size, seq_len, vocab_size]
        # targets: [batch_size, seq_len]
        loss = self.criterion(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )

        # ログ記録
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        # Learning rateのログ（オプティマイザが設定されている場合）
        opt = self.optimizers()
        if opt is not None and not isinstance(opt, list):
            self.log("learning_rate", opt.param_groups[0]["lr"], prog_bar=True)
        elif isinstance(opt, list) and len(opt) > 0:
            self.log("learning_rate", opt[0].param_groups[0]["lr"], prog_bar=True)

        # Perplexityの計算とログ
        perplexity = torch.exp(loss)
        self.log("train_perplexity", perplexity, prog_bar=False, sync_dist=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        検証ステップ

        Args:
            batch: (input_ids, targets)のタプル
            batch_idx: バッチインデックス

        Returns:
            loss: 計算された損失
        """
        input_ids, targets = batch

        # 順伝播
        logits = self(input_ids)

        # Loss計算
        loss = self.criterion(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )

        # ログ記録
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        # Perplexity
        perplexity = torch.exp(loss)
        self.log("val_perplexity", perplexity, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self) -> dict[str, Any] | Optimizer:
        """
        オプティマイザとスケジューラの設定

        Returns:
            オプティマイザとスケジューラの辞書、またはオプティマイザのみ
        """
        # Weight decay適用除外パラメータ
        no_decay = ["bias", "norm", "ln"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.optimizer_config.get("weight_decay", 0.1),
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        # オプティマイザの作成
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.optimizer_config.get("lr", 2e-4),
            betas=self.optimizer_config.get("betas", (0.9, 0.95)),
            eps=self.optimizer_config.get("eps", 1e-8),
        )

        # スケジューラの作成
        if self.scheduler_config:
            warmup_steps = self.scheduler_config.get("warmup_steps", 1000)
            max_steps = self.scheduler_config.get("max_steps", 100000)
            min_lr_ratio = self.scheduler_config.get("min_lr", 2e-5) / self.optimizer_config.get("lr", 2e-4)

            def lr_lambda(current_step: int) -> float:
                """Cosine annealing with warmup"""
                if current_step < warmup_steps:
                    # Warmup: linear increase
                    return float(current_step) / float(max(1, warmup_steps))
                else:
                    # Cosine annealing
                    progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                    return max(min_lr_ratio, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))))

            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return optimizer

    def on_train_start(self):
        """トレーニング開始時のフック"""
        # モデルサイズをログ（Wandbロガーの場合のみ）
        if self.logger is not None and hasattr(self.logger, "experiment"):
            try:
                self.logger.experiment.config.update({
                    "model_parameters": self.num_parameters,
                    "model_size_mb": self.num_parameters * 4 / 1024 / 1024,  # FP32換算
                })
            except AttributeError:
                pass  # ロガーがWandBではない場合はスキップ

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer):
        """
        勾配ゼロ化のカスタマイズ

        より効率的な set_to_none=True を使用
        """
        optimizer.zero_grad(set_to_none=True)
