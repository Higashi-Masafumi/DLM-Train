"""
Speed Monitor Callback for PyTorch Lightning

既存のSpeedMonitorFabricをPyTorch Lightning Callback形式に移行
"""

import time
from collections import deque
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only


class SpeedMonitorCallback(Callback):
    """
    トレーニング速度をモニタリングするCallback

    - Tokens/sec
    - Samples/sec (iterations/sec)
    - FLOPs utilization
    - Time per iteration

    Args:
        log_interval: ログ記録の間隔（ステップ数）
        window_size: 移動平均のウィンドウサイズ
        flops_available: 利用可能なFLOPs（理論ピーク性能）
        log_iter_throughput: イテレーション速度をログするか
        log_flops_per_sec: FLOPs/secをログするか
    """

    def __init__(
        self,
        log_interval: int = 10,
        window_size: int = 50,
        flops_available: float | None = None,
        log_iter_throughput: bool = True,
        log_flops_per_sec: bool = True,
    ):
        super().__init__()
        self.log_interval = log_interval
        self.window_size = window_size
        self.flops_available = flops_available
        self.log_iter_throughput = log_iter_throughput
        self.log_flops_per_sec = log_flops_per_sec

        # タイマー
        self.iter_times: deque[float] = deque(maxlen=window_size)
        self.token_counts: deque[int] = deque(maxlen=window_size)

        self.start_time: float | None = None
        self.last_log_time: float | None = None

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """バッチ開始時にタイマーを開始"""
        self.start_time = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """
        バッチ終了時に速度を計算してログ
        """
        if self.start_time is None:
            return

        # 経過時間を記録
        elapsed_time = time.perf_counter() - self.start_time
        self.iter_times.append(elapsed_time)

        # トークン数を記録
        # batch = (input_ids, targets)
        if isinstance(batch, (tuple, list)):
            input_ids = batch[0]
            num_tokens = input_ids.numel()
        else:
            num_tokens = batch.numel()

        self.token_counts.append(num_tokens)

        # ログ間隔でメトリクスを計算
        if (batch_idx + 1) % self.log_interval == 0:
            self._log_metrics(trainer, pl_module, batch_idx)

    @rank_zero_only
    def _log_metrics(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch_idx: int,
    ) -> None:
        """メトリクスを計算してログ"""
        if len(self.iter_times) == 0:
            return

        # 平均イテレーション時間
        avg_iter_time = sum(self.iter_times) / len(self.iter_times)

        # Tokens/sec
        total_tokens = sum(self.token_counts)
        total_time = sum(self.iter_times)
        tokens_per_sec = total_tokens / total_time if total_time > 0 else 0

        # Samples/sec
        samples_per_sec = len(self.iter_times) / total_time if total_time > 0 else 0

        # ログ記録
        metrics = {
            "speed/tokens_per_sec": tokens_per_sec,
            "speed/avg_iter_time_ms": avg_iter_time * 1000,  # ミリ秒
        }

        if self.log_iter_throughput:
            metrics["speed/iterations_per_sec"] = samples_per_sec

        # FLOPs計算（モデルパラメータ数が必要）
        if self.log_flops_per_sec and hasattr(pl_module, "num_parameters"):
            # 簡易FLOPs推定: 6 * num_params * num_tokens
            # (forward + backward passで6倍の計算が必要)
            num_params = pl_module.num_parameters
            flops_per_iter = 6 * num_params * (total_tokens / len(self.iter_times))
            flops_per_sec = flops_per_iter / avg_iter_time

            metrics["speed/tflops_per_sec"] = flops_per_sec / 1e12

            # FLOPs utilization
            if self.flops_available:
                flops_util = flops_per_sec / self.flops_available
                metrics["speed/flops_utilization"] = flops_util

        # Lightningのロガーにログ
        pl_module.log_dict(metrics, prog_bar=False, logger=True)

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """トレーニング開始時の処理"""
        self.last_log_time = time.perf_counter()

        # デバイス情報をログ
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            print(f"Training on {device_count} x {device_name}")
