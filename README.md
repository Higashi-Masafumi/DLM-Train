# 拡散言語モデルの事前学習コード

> **Note**: このリポジトリはPyTorch Lightning + Hydraベースのアーキテクチャに移行しました。
> 詳細は[移行設計書](MIGRATION_DESIGN.md)と[新しいクイックスタート](QUICKSTART_NEW.md)を参照してください。

## QuickStart

### 1. 環境セットアップ

**依存関係のインストール:**
```bash
pip install -r requirements.txt
```

**Flash Attentionのセットアップ（オプションだが推奨）:**
```bash
pip install flash-attn --no-build-isolation
```

**Hugging Faceへのログイン:**
```bash
huggingface-cli login  # meta-llama/Llama-2-7b-hfへのアクセス権限が必要
```

**Weights & Biasesへのログイン（オプション）:**
```bash
wandb login  # API keyを設定
```

### 2. データセットの準備

基本的に[TinyLlamaのデータセットセットアップ](https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md)を参考にしています。

**データセットのダウンロード（数時間かかります）:**
```bash
cd datasets
git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B
cd ..
```

**データセットの前処理（Tokenizerによる埋め込み、時間がかかります）:**
```bash
# 検証データの準備
python3 -m scripts.prepare_slim_pajama \
    --source_path datasets/SlimPajama-627B \
    --tokenizer_path meta-llama/Llama-2-7b-hf \
    --destination_path datasets/slim_star_combined \
    --split validation \
    --percentage 1.0

# 訓練データの準備
python3 -m scripts.prepare_slim_pajama \
    --source_path datasets/SlimPajama-627B \
    --tokenizer_path meta-llama/Llama-2-7b-hf \
    --destination_path datasets/slim_star_combined \
    --split train \
    --percentage 1.0
```

### 3. 事前学習の実行

**デフォルト設定で学習（6Mモデル、シングルGPU、デバッグモード）:**
```bash
python3 -m src.cli.train experiment=debug
```

**170Mモデルでスケーリング学習（推奨）:**
```bash
python3 -m src.cli.train experiment=scaling_170M
```

**カスタム設定で学習:**
```bash
# モデルサイズ、データ、トレーナー設定を個別に指定
python3 -m src.cli.train \
    model=diffusion_llama_170M \
    data=default \
    trainer=multi_gpu \
    training.flops_budget=10e18
```

**旧実装（非推奨、後方互換性のため残存）:**
```bash
python3 -m train.train --model 170 --nodes_num 1 --flops 10
```

## アーキテクチャ概要

### ディレクトリ構造

```
DLM-Train/
├── configs/                    # Hydra設定ファイル
│   ├── config.yaml            # メイン設定
│   ├── model/                 # モデル設定（6M, 170M, 1B）
│   ├── data/                  # データセット設定
│   ├── trainer/               # トレーニング設定（単一/複数GPU）
│   ├── optimizer/             # オプティマイザ設定
│   ├── scheduler/             # LRスケジューラ設定
│   ├── callbacks/             # コールバック設定
│   ├── logger/                # ロガー設定（Wandb等）
│   └── experiment/            # 実験設定（組み合わせ）
│
├── src/                       # メインソースコード
│   ├── models/                # モデル定義
│   │   ├── components/        # モデルコンポーネント
│   │   │   ├── config.py      # Pydantic設定
│   │   │   ├── rmsnorm.py     # 正規化層
│   │   │   ├── rotary_embedding.py  # RoPE
│   │   │   ├── attention.py   # アテンション層
│   │   │   ├── mlp.py         # MLP層
│   │   │   ├── block.py       # Transformerブロック
│   │   │   └── encoder.py     # TransEncoder
│   │   └── lightning_module.py  # PyTorch Lightningラッパー
│   ├── datamodules/           # データモジュール
│   │   └── slim_pajama.py     # SlimPajama DataModule
│   ├── callbacks/             # カスタムコールバック
│   │   └── speed_monitor.py   # スピードモニター
│   ├── utils/                 # ユーティリティ
│   │   └── lr_scheduler.py    # カスタムスケジューラ
│   └── cli/                   # CLIエントリーポイント
│       └── train.py           # トレーニングスクリプト
│
├── scripts/                   # データ前処理スクリプト
├── loader/                    # PackedDataset実装
└── workdir/                   # 実験結果・チェックポイント
```

### 主要な機能

- **PyTorch Lightning**: トレーニングループの自動化、分散学習、チェックポイント管理
- **Hydra**: 階層的な設定管理、実験の組み合わせ、コマンドライン上書き
- **Flash Attention v2**: 高速化されたアテンション実装
- **FSDP**: 大規模モデルの分散学習サポート
- **Wandb統合**: 実験トラッキングとメトリクス可視化

## 設定のカスタマイズ

### Hydra設定の上書き

コマンドラインから任意の設定を上書きできます:

```bash
# データバッチサイズの変更
python3 -m src.cli.train data.global_batch_size=512

# FLOPs予算の変更
python3 -m src.cli.train training.flops_budget=50e18

# 学習率の変更
python3 -m src.cli.train optimizer.lr=0.001

# 複数の設定を同時に変更
python3 -m src.cli.train \
    model=diffusion_llama_1B \
    trainer=multi_gpu \
    data.global_batch_size=1024 \
    training.flops_budget=100e18
```

### 新しい実験設定の作成

`configs/experiment/`に新しいYAMLファイルを作成:

```yaml
# configs/experiment/my_experiment.yaml
# @package _global_

defaults:
  - override /model: diffusion_llama_170M
  - override /trainer: multi_gpu
  - _self_

experiment:
  name: "my_custom_experiment"
  tags: ["custom", "170M"]

training:
  flops_budget: 20.0e18

data:
  global_batch_size: 512
```

実行:
```bash
python3 -m src.cli.train experiment=my_experiment
```

## トラブルシューティング

### 一般的な問題

1. **CUDA Out of Memory**
   - `data.micro_batch_size`を減らす
   - `trainer.accumulate_grad_batches`を増やす（勾配累積）
   - より小さいモデル（6Mまたは170M）を使用

2. **データが見つからない**
   - `data.data_dir`が正しいか確認
   - データ前処理が完了しているか確認

3. **Flash Attentionのインストール失敗**
   - CUDAバージョンを確認（11.8以上推奨）
   - `pip install flash-attn --no-build-isolation`を試す

### ログとチェックポイント

- ログファイル: `workdir/YYYY-MM-DD/HH-MM-SS/`
- チェックポイント: `workdir/YYYY-MM-DD/HH-MM-SS/checkpoints/`
- Hydra設定: `workdir/YYYY-MM-DD/HH-MM-SS/.hydra/`
- Wandbログ: `workdir/YYYY-MM-DD/HH-MM-SS/wandb/`

## 開発ガイド

### テスト実行

```bash
# 単体テスト
pytest tests/

# カバレッジ付き
pytest --cov=src tests/
```

### コードスタイル

- Python 3.9+の型ヒントを使用（`list`, `dict`, `tuple`など）
- Docstringは[Googleスタイル](https://google.github.io/styleguide/pyguide.html)を使用
- 不要なimportは削除
- Early returnを活用

## 移行ガイド

旧実装から新実装への移行については、以下のドキュメントを参照:

- [移行設計書](MIGRATION_DESIGN.md) - アーキテクチャの詳細設計
- [新クイックスタート](QUICKSTART_NEW.md) - 詳細な使用方法
- [リファクタリングレポート](REFACTORING_REPORT.md) - 移行の完了状況

## 参考資料

### 実装ベース
- [SMDM](https://github.com/ML-GSAI/SMDM) - このリポジトリのベース実装
- [TinyLlama](https://github.com/jzhang38/TinyLlama) - SMDMのベース実装
- [lit-GPT](https://github.com/Lightning-AI/litgpt) - PyTorch Lightning実装の参考

### フレームワーク
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) - トレーニングフレームワーク
- [Hydra](https://hydra.cc/) - 設定管理フレームワーク
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) - 最適化されたアテンション実装

### 論文
- [Diffusion-LM](https://arxiv.org/abs/2205.14217) - 拡散言語モデルの原論文
- [Flash Attention](https://arxiv.org/abs/2205.14135) - Flash Attentionの論文
- [FSDP](https://arxiv.org/abs/2304.11277) - Fully Sharded Data Parallel

## ライセンス

MIT License - 詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 貢献

プルリクエストを歓迎します！大きな変更の場合は、まずissueを開いて変更内容を議論してください。

## 引用

このコードを研究で使用する場合は、以下を引用してください:

```bibtex
@misc{dlm-train,
  author = {Your Name},
  title = {DLM-Train: Diffusion Language Model Pre-training},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Higashi-Masafumi/DLM-Train}
}
```
