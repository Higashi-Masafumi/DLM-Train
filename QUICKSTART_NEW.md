# PyTorch Lightning + Hydra移行 クイックスタート

このガイドでは、新しいPyTorch Lightning + Hydraベースの実装を使用して学習を開始する方法を説明します。

## 前提条件

1. **依存関係のインストール**
   ```bash
   pip install -r requirements.txt
   ```

2. **データセットの準備**（既存の方法と同じ）
   ```bash
   # データセットのダウンロード
   cd datasets
   git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B
   cd ..

   # データセットの前処理
   python3 -m scripts.prepare_slim_pajama \
       --source_path datasets/SlimPajama-627B \
       --tokenizer_path meta-llama/Llama-2-7b-hf \
       --destination_path datasets/slim_star_combined \
       --split validation \
       --percentage 1.0

   python3 -m scripts.prepare_slim_pajama \
       --source_path datasets/SlimPajama-627B \
       --tokenizer_path meta-llama/Llama-2-7b-hf \
       --destination_path datasets/slim_star_combined \
       --split train \
       --percentage 1.0
   ```

## 基本的な使用方法

### 1. デフォルト設定で学習（170Mモデル、シングルGPU）

```bash
python3 -m src.cli.train
```

### 2. 実験設定を使用（推奨）

```bash
# 170Mモデル、マルチGPU（FSDP）
python3 -m src.cli.train experiment=scaling_170M

# デバッグモード（6Mモデル、高速イテレーション）
python3 -m src.cli.train experiment=debug
```

### 3. 設定のオーバーライド

```bash
# バッチサイズとGPU数を変更
python3 -m src.cli.train \
    experiment=scaling_170M \
    data.batch_size=512 \
    trainer.devices=8

# 学習率を変更
python3 -m src.cli.train \
    experiment=scaling_170M \
    optimizer.lr=3e-4

# FLOPs予算を変更
python3 -m src.cli.train \
    experiment=scaling_170M \
    training.flops_budget=20.0e18
```

### 4. ハイパーパラメータスイープ（複数実験の自動実行）

```bash
# 学習率とバッチサイズのグリッドサーチ
python3 -m src.cli.train -m \
    experiment=scaling_170M \
    optimizer.lr=1e-4,2e-4,5e-4 \
    data.batch_size=128,256,512
```

## 設定ファイルの構造

設定ファイルは`configs/`ディレクトリに階層的に配置されています：

```
configs/
├── config.yaml              # メイン設定
├── model/                   # モデル設定
│   ├── default.yaml        # 170M (デフォルト)
│   ├── diffusion_llama_6M.yaml
│   └── diffusion_llama_1B.yaml
├── data/                    # データセット設定
│   └── default.yaml
├── trainer/                 # トレーナー設定
│   ├── default.yaml        # シングルGPU
│   └── multi_gpu.yaml      # マルチGPU (FSDP)
├── optimizer/               # オプティマイザ設定
│   └── default.yaml
├── scheduler/               # スケジューラ設定
│   └── default.yaml
├── callbacks/               # コールバック設定
│   └── default.yaml
├── logger/                  # ロガー設定
│   └── wandb.yaml
└── experiment/              # 実験設定（組み合わせ）
    ├── scaling_170M.yaml
    └── debug.yaml
```

## カスタム実験の作成

新しい実験設定を作成する場合：

```yaml
# configs/experiment/my_experiment.yaml
# @package _global_

defaults:
  - override /model: default  # または diffusion_llama_6M, diffusion_llama_1B
  - override /data: default
  - override /trainer: multi_gpu
  - override /optimizer: default
  - override /scheduler: default
  - _self_

experiment:
  name: "my_custom_experiment"
  tags: ["custom", "experiment"]
  notes: "Description of my experiment"

seed: 42

training:
  flops_budget: 10.0e18

data:
  global_batch_size: 256
  micro_batch_size: 8

optimizer:
  lr: 2.0e-4
  weight_decay: 0.1

trainer:
  devices: 4
```

実行：
```bash
python3 -m src.cli.train experiment=my_experiment
```

## チェックポイントからの再開

```bash
python3 -m src.cli.train \
    experiment=scaling_170M \
    resume_from=/path/to/checkpoint.ckpt
```

## デバッグTips

### 1. 高速デバッグモード（10イテレーションのみ実行）

```bash
python3 -m src.cli.train \
    experiment=debug \
    trainer.fast_dev_run=10
```

### 2. 設定の確認（実行せずに設定を表示）

```bash
python3 -m src.cli.train \
    experiment=scaling_170M \
    --cfg job
```

### 3. オフラインモード（Wandbなし）

```bash
python3 -m src.cli.train \
    experiment=scaling_170M \
    logger.offline=true
```

## トラブルシューティング

### Q: OOMエラーが発生する

A: `micro_batch_size`を減らすか、勾配累積を増やしてください：
```bash
python3 -m src.cli.train \
    experiment=scaling_170M \
    data.micro_batch_size=4
```

### Q: データが見つからないエラー

A: `data.data_dir`を正しいパスに設定してください：
```bash
python3 -m src.cli.train \
    experiment=scaling_170M \
    data.data_dir=/your/dataset/path
```

### Q: Hydraの出力ディレクトリを変更したい

A: `hydra.run.dir`を設定してください：
```bash
python3 -m src.cli.train \
    experiment=scaling_170M \
    hydra.run.dir=/your/output/path
```

## 既存コードとの比較

| 既存の実装 | 新しい実装 |
|-----------|-----------|
| `python3 -m train.train --model 170 --nodes_num 1 --flops 10 --batch_size 256` | `python3 -m src.cli.train experiment=scaling_170M` |
| ハードコードされた設定 | Hydraで柔軟な設定管理 |
| Fabric手動セットアップ | Lightning Trainerで自動化 |
| 独自のログ処理 | Lightning Callbacksで標準化 |

## 次のステップ

1. **新しいモデルサイズの追加**: `configs/model/`に新しいYAMLファイルを作成
2. **カスタムコールバックの追加**: `src/callbacks/`に新しいCallbackを実装
3. **新しいデータセットの追加**: `src/datamodules/`に新しいDataModuleを実装

詳細は[MIGRATION_DESIGN.md](MIGRATION_DESIGN.md)を参照してください。
