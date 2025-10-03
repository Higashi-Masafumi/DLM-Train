"""
Model Configuration classes using Pydantic

モデルアーキテクチャの設定を管理するPydanticベースの設定クラス。
型安全性と検証機能を提供します。
"""

from importlib import import_module
from typing import Any, Literal

import torch
from pydantic import (
    BaseModel,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)
from typing_extensions import Self


class Config(BaseModel):
    """
    モデル設定クラス

    Transformerベースのモデルアーキテクチャに必要な全ての設定を管理します。
    Pydanticを使用して型安全性と自動検証を提供。

    Attributes:
        name: モデル名
        block_size: 最大シーケンス長（コンテキストサイズ）
        vocab_size: 語彙サイズ
        padding_multiple: 語彙サイズのパディング単位
        padded_vocab_size: パディング後の語彙サイズ（自動計算）
        n_layer: Transformerレイヤー数
        n_head: アテンションヘッド数
        n_embd: 埋め込み次元数（hidden_size）
        rotary_percentage: Rotary Embeddingを適用する次元の割合
        parallel_residual: Parallel Residual構造を使用するか（GPT-NeoX style）
        bias: バイアス項を使用するか
        n_query_groups: Query Groupsの数（MHA/GQA/MQA）
        shared_attention_norm: Attention Normを共有するか
        norm_class_name: 正規化レイヤーのクラス名
        norm_eps: 正規化のイプシロン値
        mlp_class_name: MLPのクラス名
        intermediate_size: MLPの中間層サイズ
        condense_ratio: Diffusion用の圧縮比率
    """

    name: StrictStr = Field(description="モデル名")

    # Sequence settings
    block_size: StrictInt = Field(default=4096, description="最大シーケンス長")

    # Vocabulary settings
    vocab_size: StrictInt = Field(default=50254, description="語彙サイズ")
    padding_multiple: StrictInt = Field(default=512, description="語彙サイズのパディング単位")
    padded_vocab_size: StrictInt | None = Field(default=None, description="パディング後の語彙サイズ")

    # Architecture settings
    n_layer: StrictInt = Field(default=16, description="Transformerレイヤー数")
    n_head: StrictInt = Field(default=32, description="アテンションヘッド数")
    n_embd: StrictInt = Field(default=4096, description="埋め込み次元数")

    # Position encoding
    rotary_percentage: StrictFloat = Field(default=0.25, description="Rotary Embeddingの適用割合")

    # Architecture variants
    parallel_residual: StrictBool = Field(default=True, description="Parallel Residual構造")
    bias: StrictBool = Field(default=True, description="バイアス項を使用するか")

    # Multi-Query/Grouped-Query Attention
    # MHA: n_query_groups = n_head
    # GQA: n_query_groups = n_head // k (1 < k < n_head)
    # MQA: n_query_groups = 1
    n_query_groups: StrictInt | None = Field(default=None, description="Query Groupsの数")

    # Normalization
    shared_attention_norm: StrictBool = Field(default=False, description="Attention Normを共有")
    norm_class_name: Literal["LayerNorm", "RMSNorm", "FusedRMSNorm"] = Field(
        default="LayerNorm",
        description="正規化レイヤーのクラス名"
    )
    norm_eps: StrictFloat = Field(default=1e-5, description="正規化のイプシロン値")

    # MLP
    mlp_class_name: Literal["GptNeoXMLP", "LLaMAMLP"] = Field(
        default="GptNeoXMLP",
        description="MLPのクラス名"
    )
    intermediate_size: StrictInt | None = Field(default=None, description="MLPの中間層サイズ")

    # Diffusion specific
    condense_ratio: StrictInt = Field(default=1, description="Diffusion用の圧縮比率")

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """
        設定の検証と自動補完

        以下の処理を実行:
        - n_embd が n_head で割り切れるか確認
        - padded_vocab_size の自動計算
        - n_query_groups の自動設定
        - intermediate_size の自動設定

        Returns:
            Self: 検証済みの設定オブジェクト

        Raises:
            AssertionError: 設定が無効な場合
            ValueError: 必須パラメータが不足している場合
        """
        # n_embd が n_head で割り切れることを確認
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"

        # padded_vocab_size の自動計算
        if self.padded_vocab_size is None:
            if self.vocab_size % self.padding_multiple == 0:
                self.padded_vocab_size = self.vocab_size
            else:
                # padding_multiple の倍数に切り上げ
                self.padded_vocab_size = (
                    self.vocab_size
                    + self.padding_multiple
                    - self.vocab_size % self.padding_multiple
                )

        # n_query_groups の自動設定（MHA as default）
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0, \
                f"n_head ({self.n_head}) must be divisible by n_query_groups ({self.n_query_groups})"
        else:
            self.n_query_groups = self.n_head

        # intermediate_size の自動設定
        if self.intermediate_size is None:
            if self.mlp_class_name == "LLaMAMLP":
                raise ValueError("intermediate_size is required for LLaMAMLP")
            # デフォルト: 4倍
            self.intermediate_size = self.n_embd * 4

        return self

    @property
    def head_size(self) -> int:
        """
        各アテンションヘッドのサイズ

        Returns:
            int: head_size = n_embd // n_head
        """
        return self.n_embd // self.n_head

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        """
        事前定義された設定から Config を作成

        Args:
            name: 事前定義された設定名（e.g., "Llama-170M"）
            **kwargs: 追加でオーバーライドするパラメータ

        Returns:
            Self: 作成された Config オブジェクト

        Raises:
            KeyError: 指定された名前の設定が存在しない場合

        Example:
            >>> config = Config.from_name("Llama-170M", block_size=4096)
        """
        if name not in name_to_config:
            raise KeyError(f"Unknown config name: {name}")

        conf_dict = name_to_config[name].copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @property
    def mlp_class(self):
        """
        MLPクラスを動的にインポート

        Returns:
            Type: MLPクラス
        """
        # 循環インポートを避けるため、動的インポート
        from . import mlp as mlp_module
        return getattr(mlp_module, self.mlp_class_name)

    @property
    def norm_class(self):
        """
        正規化クラスを動的にインポート

        Returns:
            Type: 正規化クラス
        """
        if self.norm_class_name in ("RMSNorm", "FusedRMSNorm"):
            from .rmsnorm import FusedRMSNorm, RMSNorm
            return RMSNorm if self.norm_class_name == "RMSNorm" else FusedRMSNorm

        # PyTorchの標準クラス
        return getattr(torch.nn, self.norm_class_name)


# ─────────────────────────────────────────────────────────────────────────────
# 事前定義された設定
# ─────────────────────────────────────────────────────────────────────────────

# Diffusion LLaMA configurations
Diff_LLaMA = [
    dict(
        name="Diff_LLaMA_6M",
        block_size=2048,
        vocab_size=32000,
        n_layer=4,
        n_head=4,
        n_embd=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="FusedRMSNorm",
        norm_eps=1e-5,
        mlp_class_name="LLaMAMLP",
        intermediate_size=688,
        condense_ratio=1,
    ),
    dict(
        name="Diff_LLaMA_19M",
        block_size=2048,
        vocab_size=32000,
        n_layer=6,
        n_head=6,
        n_embd=384,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="FusedRMSNorm",
        norm_eps=1e-5,
        mlp_class_name="LLaMAMLP",
        intermediate_size=1024,
        condense_ratio=1,
    ),
    dict(
        name="Diff_LLaMA_170M",
        block_size=2048,
        vocab_size=32000,
        n_layer=12,
        n_head=12,
        n_embd=768,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="FusedRMSNorm",
        norm_eps=1e-5,
        mlp_class_name="LLaMAMLP",
        intermediate_size=2048,
        condense_ratio=1,
    ),
    dict(
        name="Diff_LLaMA_1B",
        block_size=2048,
        vocab_size=32000,
        n_layer=22,
        n_head=16,
        n_embd=2048,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="FusedRMSNorm",
        norm_eps=1e-5,
        mlp_class_name="LLaMAMLP",
        intermediate_size=5504,
        condense_ratio=1,
    ),
]

# すべての設定を収集
configs = []
configs.extend(Diff_LLaMA)

# 名前からConfigへのマッピング
name_to_config = {config["name"]: config for config in configs}

# エクスポート
__all__ = ["Config", "configs", "name_to_config"]
