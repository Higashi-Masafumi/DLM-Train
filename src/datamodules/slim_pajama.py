"""
PyTorch Lightning DataModule for SlimPajama Dataset

PackedDatasetを使用した効率的なデータローディングを提供します。
"""

from pathlib import Path
from typing import Optional

import lightning as L
import torch
from torch.utils.data import DataLoader

from loader.packed_dataset import CombinedDataset, PackedDataset


def collate_fn(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """PackedDatasetのバッチを(input_ids, targets)のタプルに変換.

    PackedDatasetは block_size 長のトークン列を返します。
    これを input_ids[:-1] と targets[1:] に分割して、
    言語モデルの入力と目標値にします。

    Args:
        batch: shape (batch_size, block_size) のトークン列のリスト

    Returns:
        (input_ids, targets) のタプル
        - input_ids: shape (batch_size, block_size-1)
        - targets: shape (batch_size, block_size-1)
    """
    # バッチをスタック
    x = torch.stack(batch)  # (batch_size, block_size)

    # 入力と目標値に分割
    input_ids = x[:, :-1].contiguous()  # (batch_size, block_size-1)
    targets = x[:, 1:].contiguous()  # (batch_size, block_size-1)

    return input_ids, targets


class SlimPajamaDataModule(L.LightningDataModule):
    """
    SlimPajama Dataset用のLightning DataModule

    既存のPackedDatasetを活用し、train/valデータローダーを提供します。

    Args:
        data_dir: データセットのルートディレクトリ
        train_prefix: 訓練データのプレフィックス
        val_prefix: 検証データのプレフィックス
        global_batch_size: グローバルバッチサイズ（全デバイス合計）
        micro_batch_size: マイクロバッチサイズ（1デバイスあたり）
        num_workers: DataLoaderのワーカー数
        block_size: トークンシーケンス長
        train_chunks: 訓練データのチャンク数
        val_chunks: 検証データのチャンク数
        shuffle: データをシャッフルするか
        seed: ランダムシード
        pin_memory: pin_memoryを有効化するか
        persistent_workers: persistent_workersを有効化するか
    """

    def __init__(
        self,
        data_dir: str | Path,
        train_prefix: str = "train_slimpajama",
        val_prefix: str = "validation",
        global_batch_size: int = 256,
        micro_batch_size: int = 8,
        num_workers: int = 4,
        block_size: int = 2048,
        train_chunks: int = 100,
        val_chunks: int = 10,
        shuffle: bool = True,
        seed: int = 42,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        super().__init__()

        # パラメータを保存
        self.data_dir = Path(data_dir)
        self.train_prefix = train_prefix
        self.val_prefix = val_prefix
        self.global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.block_size = block_size
        self.train_chunks = train_chunks
        self.val_chunks = val_chunks
        self.shuffle = shuffle
        self.seed = seed
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        # データセット（setupで初期化）
        self.train_dataset: Optional[PackedDataset] = None
        self.val_dataset: Optional[PackedDataset] = None

    def prepare_data(self):
        """
        データの準備（ダウンロード・前処理など）

        注: この処理はrank 0でのみ実行されます
        分散学習時に複数プロセスで同時実行されることを避けるため、
        データの前処理スクリプト（scripts/prepare_slim_pajama.py）は
        事前に実行しておくことを推奨します。
        """
        # データが既に存在することを確認
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"Please run data preparation script first:\n"
                f"python -m scripts.prepare_slim_pajama --destination_path {self.data_dir}"
            )

    def setup(self, stage: str):
        """
        データセットのセットアップ

        Args:
            stage: 'fit', 'validate', 'test', 'predict'のいずれか
        """
        if stage == "fit" or stage is None:
            # 訓練データセットの構築
            train_files = self._get_data_files(self.train_prefix)
            self.train_dataset = PackedDataset(
                filenames=train_files,
                n_chunks=self.train_chunks,
                block_size=self.block_size + 1,  # +1 for target
                seed=self.seed,
                shuffle=self.shuffle,
                wrap=True,  # データを繰り返し使用
            )

            # 検証データセットの構築
            val_files = self._get_data_files(self.val_prefix)
            self.val_dataset = PackedDataset(
                filenames=val_files,
                n_chunks=self.val_chunks,
                block_size=self.block_size + 1,
                seed=self.seed,
                shuffle=False,  # 検証データはシャッフルしない
                wrap=False,
            )

    def _get_data_files(self, prefix: str) -> list[Path]:
        """
        指定されたプレフィックスのデータファイルを取得

        Args:
            prefix: ファイル名のプレフィックス

        Returns:
            データファイルのパスリスト
        """
        # .binファイルを検索
        pattern = f"{prefix}_*.bin"
        files = sorted(self.data_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(
                f"No data files found with pattern: {self.data_dir / pattern}"
            )

        return files

    def train_dataloader(self) -> DataLoader:
        """訓練用DataLoader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.micro_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """検証用DataLoader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.micro_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=collate_fn,
        )

    def teardown(self, stage: str):
        """
        クリーンアップ処理

        Args:
            stage: 'fit', 'validate', 'test', 'predict'のいずれか
        """
        # 必要に応じてリソースの解放
        pass
