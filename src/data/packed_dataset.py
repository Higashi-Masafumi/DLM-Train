"""
Packed Dataset Implementation

効率的なデータローディングのためのPackedDataset実装。
Fairseq, Megatron-LMのindexed_datasetに触発されています。

Reference:
    https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/indexed_dataset.py
"""

import os
import random
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

# データ型マッピング
dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}


def code(dtype: np.dtype) -> int:
    """numpy dtypeをコードに変換.

    Args:
        dtype: numpy dtype

    Returns:
        dtype code

    Raises:
        ValueError: サポートされていないdtypeの場合
    """
    for k, v in dtypes.items():
        if v == dtype:
            return k
    raise ValueError(f"Unsupported dtype: {dtype}")


# ファイルヘッダーの定義
HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24  # bytes


class PackedDataset(IterableDataset):
    """
    効率的なトークンシーケンスのためのPacked Dataset

    複数の.binファイルから効率的にデータを読み込みます。
    メモリマップを使用してメモリ効率を向上させています。

    Args:
        filenames: データファイルのパスリスト
        n_chunks: 一度にロードするチャンク数
        block_size: 各ブロックのトークン数
        seed: ランダムシード
        shuffle: データをシャッフルするか
        wrap: データセットの終端で先頭に戻るか
        num_processes: プロセス数（分散学習用）
        process_rank: プロセスランク（分散学習用）

    Example:
        >>> files = list(Path("data").glob("train_*.bin"))
        >>> dataset = PackedDataset(
        ...     filenames=files,
        ...     n_chunks=10,
        ...     block_size=2048,
        ...     shuffle=True
        ... )
        >>> for tokens in dataset:
        ...     print(tokens.shape)  # torch.Size([2048])
    """

    def __init__(
        self,
        filenames: list[str] | list[Path],
        n_chunks: int,
        block_size: int,
        seed: int = 12345,
        shuffle: bool = True,
        wrap: bool = False,
        num_processes: int = 1,
        process_rank: int = 0,
    ):
        self._filenames = [Path(f) if isinstance(f, str) else f for f in filenames]
        self._n_chunks = n_chunks
        self._block_size = block_size
        self._seed = seed
        self._shuffle = shuffle
        self._wrap = wrap
        self._num_processes = num_processes
        self._process_rank = process_rank

    def __iter__(self):
        """イテレータを返す."""
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id

        max_num_files = len(self._filenames) // num_shards * num_shards
        filenames = self._filenames[shard_id:max_num_files:num_shards]

        return PackedDatasetIterator(
            filenames=filenames,
            n_chunks=self._n_chunks,
            block_size=self._block_size,
            seed=self._seed,
            shuffle=self._shuffle,
            wrap=self._wrap,
        )


class PackedDatasetBuilder:
    """
    PackedDatasetを構築するためのビルダークラス

    トークン配列を効率的なバイナリ形式で保存します。

    Args:
        outdir: 出力ディレクトリ
        prefix: ファイル名のプレフィックス
        chunk_size: 各チャンクのトークン数
        sep_token: 区切りトークン
        dtype: データ型（"auto"の場合はvocab_sizeから自動決定）
        vocab_size: 語彙サイズ（dtype="auto"の場合に必要）

    Example:
        >>> builder = PackedDatasetBuilder(
        ...     outdir="data",
        ...     prefix="train",
        ...     chunk_size=1024*1024,
        ...     sep_token=0,
        ...     vocab_size=32000
        ... )
        >>> tokens = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
        >>> builder.add_array(tokens)
        >>> builder.write_reminder()
    """

    def __init__(
        self,
        outdir: str | Path,
        prefix: str,
        chunk_size: int,
        sep_token: int,
        dtype: str | np.dtype = "auto",
        vocab_size: int | None = None,
    ):
        if dtype == "auto":
            if vocab_size is None:
                raise ValueError("vocab_size cannot be None when dtype='auto'")
            # vocab_size < 65500ならuint16、それ以外はint32
            self._dtype: np.dtype = np.dtype(np.uint16 if vocab_size < 65500 else np.int32)
        else:
            self._dtype: np.dtype = np.dtype(dtype) if isinstance(dtype, str) else dtype

        self._counter = 0
        self._chunk_size = chunk_size
        self._outdir = Path(outdir)
        self._prefix = prefix
        self._sep_token = sep_token
        self._arr = np.zeros(self._chunk_size, dtype=self._dtype)
        self._arr.fill(self._sep_token)
        self._idx = 0
        self._version = 1
        self._filenames: list[str] = []

    def _write_chunk(self) -> None:
        """現在のチャンクをファイルに書き込む."""
        filename = f"{self._prefix}_{self._counter:010d}.bin"
        filepath = self._outdir / filename

        with open(filepath, "wb") as f:
            f.write(HDR_MAGIC)
            f.write(struct.pack("<Q", self._version))
            f.write(struct.pack("<B", code(self._dtype)))
            f.write(struct.pack("<Q", self._chunk_size))
            f.write(self._arr.tobytes(order="C"))

        self._filenames.append(str(filepath))
        self._counter += 1
        self._arr.fill(self._sep_token)
        self._idx = 0

    @property
    def dtype(self) -> np.dtype:
        """データ型を返す."""
        return self._dtype

    @property
    def filenames(self) -> list[str]:
        """作成されたファイル名のリストを返す."""
        return self._filenames.copy()

    def add_array(self, arr: np.ndarray) -> None:
        """トークン配列を追加.

        Args:
            arr: 追加するトークン配列
        """
        while self._idx + arr.shape[0] > self._chunk_size:
            part_len = self._chunk_size - self._idx
            self._arr[self._idx : self._idx + part_len] = arr[:part_len]
            self._write_chunk()
            arr = arr[part_len:]

        arr_len = arr.shape[0]
        self._arr[self._idx : self._idx + arr_len] = arr
        self._idx += arr_len

    def write_reminder(self) -> None:
        """残りのデータを書き込む."""
        self._write_chunk()


class PackedDatasetIterator:
    """
    PackedDatasetのイテレータ

    メモリマップを使用して効率的にデータを読み込みます。
    """

    def __init__(
        self,
        filenames: list[Path],
        n_chunks: int,
        block_size: int,
        seed: int,
        shuffle: bool,
        wrap: bool,
    ):
        self._seed = seed
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed) if shuffle else None
        self._block_idxs = None

        self._wrap = wrap
        self._filenames = filenames
        self._file_idx = 0

        self._n_chunks = n_chunks

        self._dtype: np.dtype | None = None
        self._block_size = block_size
        self._n_blocks: int | None = None

        self._mmaps: list[np.memmap] = []
        self._buffers: list[memoryview] = []

        self._block_idxs: np.ndarray | range | None = None
        self._curr_idx = 0

        self._load_n_chunks()

    def _read_header(self, path: Path) -> tuple[np.dtype, int]:
        """ファイルヘッダーを読み込む.

        Args:
            path: ファイルパス

        Returns:
            (dtype, chunk_size)のタプル

        Raises:
            AssertionError: ヘッダーが不正な場合
        """
        with open(path, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            assert magic == HDR_MAGIC, "File doesn't match expected format."
            version = struct.unpack("<Q", f.read(8))
            assert version == (1,), f"Unsupported version: {version}"
            (dtype_code,) = struct.unpack("<B", f.read(1))
            dtype = dtypes[dtype_code]
            (chunk_size,) = struct.unpack("<Q", f.read(8))
        return dtype, chunk_size

    def _close_mmaps(self) -> None:
        """メモリマップを閉じる."""
        for mmap in self._mmaps:
            del mmap

    def _load_n_chunks(self) -> None:
        """n_chunks個のチャンクをロードする."""
        self._close_mmaps()
        self._mmaps = []
        self._buffers = []

        if self._n_chunks > len(self._filenames[self._file_idx :]):
            self._file_idx = 0

        for i in range(self._n_chunks):
            filename = self._filenames[self._file_idx + i]
            if self._dtype is None:
                self._dtype, self._chunk_size = self._read_header(filename)
                self._n_blocks = self._chunk_size // self._block_size

            mmap = np.memmap(filename, mode="r", order="C", offset=HDR_SIZE)
            self._mmaps.append(mmap)
            self._buffers.append(memoryview(mmap))  # type: ignore

        self._file_idx += self._n_chunks
        assert self._n_blocks is not None, "n_blocks should be initialized"
        n_all_blocks = self._n_chunks * self._n_blocks

        assert self._rng is not None or not self._shuffle
        self._block_idxs = (
            self._rng.permutation(n_all_blocks) if self._shuffle and self._rng is not None else range(n_all_blocks)
        )

        self._curr_idx = 0

    def __del__(self):
        """デストラクタ: メモリマップをクリーンアップ."""
        self._close_mmaps()
        del self._mmaps
        del self._buffers

    def __iter__(self):
        """イテレータを返す."""
        return self

    def __next__(self) -> torch.Tensor:
        """次のブロックを返す.

        Returns:
            shape (block_size,) のトークンテンソル

        Raises:
            StopIteration: データセットの終端に達した場合
        """
        assert self._block_idxs is not None, "block_idxs should be initialized"
        if self._curr_idx >= len(self._block_idxs):
            self._load_n_chunks()

        assert self._block_idxs is not None
        block_idx = self._block_idxs[self._curr_idx]
        assert self._n_blocks is not None
        chunk_id = block_idx // self._n_blocks
        buffer = self._buffers[chunk_id]
        elem_id = (block_idx % self._n_blocks) * self._block_size
        assert self._dtype is not None
        offset = np.dtype(self._dtype).itemsize * elem_id
        arr = np.frombuffer(
            buffer, dtype=self._dtype, count=self._block_size, offset=offset
        )
        self._curr_idx += 1
        return torch.from_numpy(arr.astype(np.int64))


class CombinedDataset(IterableDataset):
    """
    複数のデータセットを組み合わせるデータセット

    重み付けサンプリングで複数のデータセットからデータを取得します。

    Args:
        datasets: データセットのリスト
        seed: ランダムシード
        weights: 各データセットの重み（Noneの場合は均等）

    Example:
        >>> dataset1 = PackedDataset(files1, n_chunks=10, block_size=2048)
        >>> dataset2 = PackedDataset(files2, n_chunks=10, block_size=2048)
        >>> combined = CombinedDataset(
        ...     datasets=[dataset1, dataset2],
        ...     weights=[0.7, 0.3],
        ...     seed=42
        ... )
    """

    def __init__(
        self,
        datasets: list[IterableDataset],
        seed: int,
        weights: list[float] | None = None,
    ):
        self._seed = seed
        self._datasets = datasets
        n_datasets = len(datasets)
        self._weights = weights if weights is not None else [1 / n_datasets] * n_datasets

    def __iter__(self):
        """イテレータを返す."""
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)


class CombinedDatasetIterator:
    """
    CombinedDatasetのイテレータ

    重み付けランダムサンプリングでデータセットを選択します。
    """

    def __init__(
        self,
        datasets: list[IterableDataset],
        seed: int,
        weights: list[float],
    ):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __iter__(self):
        """イテレータを返す（自分自身）."""
        return self

    def __next__(self) -> torch.Tensor:
        """次のサンプルを返す.

        Returns:
            重み付けサンプリングで選択されたデータセットからのトークンテンソル
        """
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        return next(dataset)
