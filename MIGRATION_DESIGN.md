# PyTorch Lightning + Hydra 移行設計書

## 目次
1. [概要](#概要)
2. [新しいディレクトリ構造](#新しいディレクトリ構造)
3. [主要コンポーネントの設計](#主要コンポーネントの設計)
4. [Hydra設定ファイルの構造](#hydra設定ファイルの構造)
5. [移行手順](#移行手順)
6. [メリット](#メリット)

## 概要

### 移行の目的
- **設定管理の改善**: Hydraで階層的な設定管理を実現
- **コードの再利用性向上**: PyTorch Lightningでボイラープレートを削減
- **実験管理の簡素化**: Wandb統合とチェックポイント管理の自動化
- **スケーラビリティ**: マルチGPU・マルチノード学習の簡素化
- **メンテナンス性**: 関心の分離と明確なインターフェース

### 現状の課題
1. **ハードコードされた設定**: `train.py`内に多数のハイパーパラメータが直接記述
2. **重複コード**: `train.py`と`train_mdm.py`で似た処理が重複
3. **設定の柔軟性不足**: 実験ごとに異なる設定を管理しにくい
4. **スケーリングロジックの散在**: バッチサイズやステップ数の計算が複数箇所に分散

## 新しいディレクトリ構造

```
DLM-Train/
├── configs/                    # Hydra設定ファイル
│   ├── config.yaml            # メイン設定
│   ├── model/                 # モデル設定
│   │   ├── diffusion_llama_6M.yaml
│   │   ├── diffusion_llama_170M.yaml
│   │   ├── diffusion_llama_1B.yaml
│   │   └── default.yaml
│   ├── data/                  # データセット設定
│   │   ├── slim_pajama.yaml
│   │   ├── slim_star.yaml
│   │   └── default.yaml
│   ├── trainer/               # トレーニング設定
│   │   ├── default.yaml
│   │   ├── single_gpu.yaml
│   │   ├── multi_gpu.yaml
│   │   └── multi_node.yaml
│   ├── optimizer/             # オプティマイザ設定
│   │   ├── adamw.yaml
│   │   └── default.yaml
│   ├── scheduler/             # スケジューラ設定
│   │   ├── cosine.yaml
│   │   └── default.yaml
│   └── experiment/            # 実験設定（組み合わせ）
│       ├── scaling_6M.yaml
│       ├── scaling_170M.yaml
│       └── debug.yaml
│
├── src/                       # メインソースコード
│   ├── __init__.py
│   ├── datamodules/           # LightningDataModule
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── packed_dataset.py
│   │   └── slim_pajama.py
│   ├── models/                # LightningModule & モデル定義
│   │   ├── __init__.py
│   │   ├── lightning_module.py      # PyTorch Lightning wrapper
│   │   ├── diffusion_module.py      # Diffusion LM module
│   │   ├── config.py                # 既存のConfig（一部移行）
│   │   ├── diffusionlm.py           # 既存のTransEncoder
│   │   ├── model.py                 # 既存のGPT
│   │   ├── rmsnorm.py
│   │   ├── fused_rotary_embedding.py
│   │   └── tokenizer.py
│   ├── callbacks/             # PyTorch Lightning callbacks
│   │   ├── __init__.py
│   │   ├── speed_monitor.py         # 既存のSpeedMonitorを移行
│   │   ├── checkpoint.py
│   │   └── gradient_monitor.py
│   ├── utils/                 # ユーティリティ
│   │   ├── __init__.py
│   │   ├── utils.py                 # 既存のutilsを整理
│   │   └── lr_scheduler.py
│   └── cli/                   # CLIエントリーポイント
│       ├── __init__.py
│       ├── train.py                 # メイントレーニングスクリプト
│       └── prepare_data.py          # データ前処理スクリプト
│
├── scripts/                   # 補助スクリプト
│   ├── prepare_slim_pajama.py      # 既存のまま
│   └── convert_checkpoint.py       # チェックポイント変換用
│
├── tests/                     # テストコード
│   ├── __init__.py
│   ├── test_datamodules.py
│   ├── test_models.py
│   └── test_training.py
│
├── loader/                    # 既存のデータローダー（段階的に移行）
│   └── packed_dataset.py
│
├── workdir/                   # 実験結果・チェックポイント
│   └── .gitkeep
│
├── requirements.txt
├── setup.py                   # パッケージ化用
├── README.md
├── MIGRATION_DESIGN.md        # このドキュメント
└── .gitignore
```

## 主要コンポーネントの設計

### 1. LightningModule（`src/models/lightning_module.py`）

```python
class DiffusionLMLightningModule(L.LightningModule):
    """
    Diffusion Language Model用のPyTorch Lightning Module

    責務:
    - モデルの初期化
    - 学習・検証ステップの定義
    - オプティマイザ・スケジューラの設定
    - メトリクスのログ記録
    """

    def __init__(self, config: Config, optimizer_config: dict, scheduler_config: dict):
        pass

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """既存のTransEncoderを呼び出し"""
        pass

    def training_step(self, batch, batch_idx):
        """学習ステップ: loss計算とログ記録"""
        pass

    def validation_step(self, batch, batch_idx):
        """検証ステップ"""
        pass

    def configure_optimizers(self):
        """オプティマイザとスケジューラの設定（Hydraから注入）"""
        pass
```

### 2. LightningDataModule（`src/datamodules/slim_pajama.py`）

```python
class SlimPajamaDataModule(L.LightningDataModule):
    """
    SlimPajamaデータセット用のDataModule

    責務:
    - データセットの準備
    - train/val DataLoaderの提供
    - データ前処理の管理
    """

    def __init__(self, data_dir: Path, batch_size: int, num_workers: int, ...):
        pass

    def prepare_data(self):
        """データのダウンロード・前処理（rank 0のみ実行）"""
        pass

    def setup(self, stage: str):
        """データセットの構築"""
        pass

    def train_dataloader(self):
        """訓練用DataLoader"""
        pass

    def val_dataloader(self):
        """検証用DataLoader"""
        pass
```

### 3. メイントレーニングスクリプト（`src/cli/train.py`）

```python
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """
    メイントレーニング関数

    Hydraが設定を自動的に読み込み、オーバーライドを適用
    """

    # シード設定
    L.seed_everything(cfg.seed)

    # DataModule初期化
    datamodule = hydra.utils.instantiate(cfg.data)

    # Model初期化
    model = hydra.utils.instantiate(cfg.model)

    # Callbacks初期化
    callbacks = [hydra.utils.instantiate(cb) for cb in cfg.callbacks.values()]

    # Logger初期化
    logger = hydra.utils.instantiate(cfg.logger)

    # Trainer初期化
    trainer = L.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        logger=logger
    )

    # トレーニング開始
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
```

## Hydra設定ファイルの構造

### メイン設定（`configs/config.yaml`）

```yaml
# @package _global_

defaults:
  - model: default
  - data: default
  - trainer: default
  - optimizer: default
  - scheduler: default
  - callbacks: default
  - logger: wandb
  - experiment: null  # 実験設定で上書き可能
  - _self_

# 基本設定
seed: 42
output_dir: ${hydra:runtime.cwd}/workdir/${now:%Y-%m-%d}/${now:%H-%M-%S}

# Hydra runtime設定
hydra:
  run:
    dir: ${output_dir}
  job:
    chdir: true
```

### モデル設定（`configs/model/diffusion_llama_170M.yaml`）

```yaml
# @package _global_

_target_: src.models.lightning_module.DiffusionLMLightningModule

model_config:
  name: "Llama-170M"
  block_size: 2048
  vocab_size: 32000
  n_layer: 12
  n_head: 12
  n_embd: 768
  rotary_percentage: 1.0
  parallel_residual: false
  bias: false
  norm_class_name: "FusedRMSNorm"
  norm_eps: 1e-5
  mlp_class_name: "LLaMAMLP"
  intermediate_size: 2048

optimizer_config:
  lr: 2.0e-4
  weight_decay: 0.1
  betas: [0.9, 0.95]

scheduler_config:
  warmup_steps: ${training.max_steps}  # 自動計算
  max_steps: ${training.max_steps}
  min_lr: 2.0e-5
```

### データ設定（`configs/data/slim_pajama.yaml`）

```yaml
# @package _global_

_target_: src.datamodules.slim_pajama.SlimPajamaDataModule

data_dir: /work/datasets/slim_star_combined
train_prefix: "train_slimpajama"
val_prefix: "validation"
batch_size: 256  # グローバルバッチサイズ
micro_batch_size: 8
num_workers: 4
block_size: 2048
train_chunks: 100
val_chunks: 10
```

### トレーナー設定（`configs/trainer/multi_gpu.yaml`）

```yaml
# @package _global_

accelerator: gpu
devices: 4
strategy: fsdp
precision: bf16-mixed

max_steps: ${training.max_steps}  # 実験設定から計算
val_check_interval: 5000
log_every_n_steps: 10
gradient_clip_val: 1.0

accumulate_grad_batches: ${training.gradient_accumulation_steps}

# FSDP設定
strategy_config:
  auto_wrap_policy: "transformer_auto_wrap_policy"
  activation_checkpointing: true
```

### 実験設定（`configs/experiment/scaling_170M.yaml`）

```yaml
# @package _global_

defaults:
  - override /model: diffusion_llama_170M
  - override /data: slim_pajama
  - override /trainer: multi_gpu
  - override /optimizer: adamw
  - override /scheduler: cosine

# 実験固有の設定
experiment_name: "scaling_170M_slimpajama"
seed: 42

# FLOPsベースの学習ステップ計算
training:
  flops_budget: 10.0e18  # 10e18 FLOPs
  model_params: 169.897728e6  # 170M
  tokens_per_batch: ${data.batch_size} * ${data.block_size}  # 256 * 2048
  max_steps: ???  # 自動計算される

# データ設定のオーバーライド
data:
  batch_size: 256
  micro_batch_size: 8

# ロガー設定
logger:
  project: "diffusion-lm-scaling"
  name: ${experiment_name}
  tags: ["scaling", "170M", "slimpajama"]
```

### Callbacks設定（`configs/callbacks/default.yaml`）

```yaml
# @package _global_

callbacks:
  model_checkpoint:
    _target_: lightning.pytorch.callbacks.ModelCheckpoint
    dirpath: ${output_dir}/checkpoints
    filename: "{epoch:02d}-{step:06d}-{val_loss:.4f}"
    monitor: val_loss
    mode: min
    save_top_k: 3
    save_last: true
    every_n_train_steps: 5000

  speed_monitor:
    _target_: src.callbacks.speed_monitor.SpeedMonitorCallback
    log_interval: 10
    flops_available: ${training.flops_budget}

  learning_rate_monitor:
    _target_: lightning.pytorch.callbacks.LearningRateMonitor
    logging_interval: step
    log_momentum: true

  gradient_monitor:
    _target_: src.callbacks.gradient_monitor.GradientMonitor
    log_interval: 100
```

## 移行手順

### Phase 1: 準備フェーズ（1-2日）

1. **依存関係の追加**
   ```bash
   # requirements.txtに追加
   hydra-core>=1.3.0
   pytorch-lightning>=2.0.0
   ```

2. **ディレクトリ構造の作成**
   ```bash
   mkdir -p configs/{model,data,trainer,optimizer,scheduler,experiment,callbacks}
   mkdir -p src/{datamodules,models,callbacks,utils,cli}
   mkdir -p tests
   ```

3. **基本設定ファイルの作成**
   - `configs/config.yaml`
   - `configs/model/default.yaml`
   - `configs/data/default.yaml`

### Phase 2: コアコンポーネントの移行（3-5日）

4. **LightningDataModuleの実装**
   - `src/datamodules/base.py`: ベースクラス
   - `src/datamodules/slim_pajama.py`: SlimPajama用DataModule
   - 既存の`loader/packed_dataset.py`を活用

5. **LightningModuleの実装**
   - `src/models/lightning_module.py`: メインLightningModule
   - 既存の`models/diffusionlm.py`のTransEncoderをラップ
   - `training_step`, `validation_step`の実装
   - `configure_optimizers`の実装

6. **Callbacksの移行**
   - `src/callbacks/speed_monitor.py`: 既存のSpeedMonitorを移行
   - `src/callbacks/checkpoint.py`: カスタムチェックポイントロジック

### Phase 3: CLIとHydra統合（2-3日）

7. **メイントレーニングスクリプトの作成**
   - `src/cli/train.py`: Hydra統合メインスクリプト
   - 既存の`train/train.py`のロジックを移行

8. **設定ファイルの実装**
   - モデルサイズごとの設定ファイル
   - データセット設定
   - トレーナー設定（single/multi GPU/node）

### Phase 4: テストと検証（2-3日）

9. **単体テストの作成**
   - DataModuleのテスト
   - LightningModuleのテスト
   - Callbacksのテスト

10. **統合テスト**
    - 小規模データセットでの動作確認
    - シングルGPUでの学習
    - マルチGPUでの学習

11. **既存スクリプトとの結果比較**
    - 同じ設定で既存コードと新コードを実行
    - Loss曲線、学習速度の比較

### Phase 5: ドキュメント整備と移行完了（1-2日）

12. **ドキュメント作成**
    - README.mdの更新
    - 使用例の追加
    - トラブルシューティングガイド

13. **クリーンアップ**
    - 旧コードのアーカイブ（`legacy/`ディレクトリへ移動）
    - 不要なファイルの削除

## メリット

### 1. 設定管理の改善
- **階層的な設定**: モデル、データ、トレーニングなどを独立して管理
- **コマンドラインオーバーライド**: `python train.py model=diffusion_llama_170M data.batch_size=512`
- **実験の再現性**: 設定ファイルで完全な実験設定を保存

### 2. コードの簡潔化
- **ボイラープレート削減**: Lightning Trainerが自動処理
  - 分散学習のセットアップ
  - チェックポイント管理
  - ロギング
  - 勾配クリッピング
- **関心の分離**: データ、モデル、トレーニングロジックが明確に分離

### 3. 実験管理の簡素化
```bash
# 異なるモデルサイズの実験
python src/cli/train.py experiment=scaling_6M
python src/cli/train.py experiment=scaling_170M
python src/cli/train.py experiment=scaling_1B

# ハイパーパラメータスイープ
python src/cli/train.py -m optimizer.lr=1e-4,2e-4,5e-4 data.batch_size=128,256,512
```

### 4. スケーラビリティ
- **自動分散学習**: FSDPやDDPの設定がシンプル
- **マルチノード対応**: 設定ファイルで簡単に切り替え
- **混合精度学習**: `precision=bf16-mixed`で簡単に有効化

### 5. デバッグとモニタリング
- **組み込みプロファイラ**: Lightning Profiler
- **進捗バー**: tqdmベースの自動進捗表示
- **Early Stopping**: コールバックで簡単に実装

## 使用例

### 基本的な使用方法

```bash
# デフォルト設定で学習
python src/cli/train.py

# 特定の実験設定を使用
python src/cli/train.py experiment=scaling_170M

# 設定をオーバーライド
python src/cli/train.py \
    experiment=scaling_170M \
    data.batch_size=512 \
    trainer.devices=8 \
    optimizer.lr=3e-4

# デバッグモード（小規模データ、高速イテレーション）
python src/cli/train.py \
    experiment=debug \
    trainer.fast_dev_run=10
```

### ハイパーパラメータスイープ

```bash
# Hydraのマルチラン機能
python src/cli/train.py -m \
    experiment=scaling_170M \
    optimizer.lr=1e-4,2e-4,5e-4 \
    data.batch_size=128,256
```

### チェックポイントからの再開

```bash
python src/cli/train.py \
    experiment=scaling_170M \
    trainer.resume_from_checkpoint=/path/to/checkpoint.ckpt
```

## 注意点とベストプラクティス

### 1. 段階的な移行
- 一度に全てを移行せず、コンポーネントごとに段階的に移行
- 各フェーズで既存コードとの動作を比較・検証

### 2. 既存コードの保持
- 移行完了まで既存コードを`legacy/`ディレクトリで保持
- 問題があれば既存コードに戻れるようにする

### 3. テストの重要性
- 移行前後で数値的な一致を確認
- 単体テストと統合テストを十分に実施

### 4. ドキュメント
- 各設定ファイルにコメントを記載
- READMEに使用例を充実させる

## まとめ

この移行により、以下が実現されます:

1. **保守性の向上**: 明確な責務分離と設定管理
2. **実験の効率化**: Hydraによる柔軟な設定管理
3. **スケーラビリティ**: Lightning Trainerの自動分散学習
4. **再現性**: 設定ファイルによる完全な実験記録
5. **拡張性**: 新しいモデルやデータセットの追加が容易

移行は約2週間程度を想定していますが、チームの状況に応じて調整可能です。
