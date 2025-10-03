"""
メイントレーニングスクリプト (Hydra + PyTorch Lightning)

使用例:
    # デフォルト設定で実行
    python -m src.cli.train

    # 特定の実験設定を使用
    python -m src.cli.train experiment=scaling_170M

    # 設定をオーバーライド
    python -m src.cli.train experiment=scaling_170M data.batch_size=512 trainer.devices=8

    # デバッグモード
    python -m src.cli.train experiment=debug

    # ハイパーパラメータスイープ
    python -m src.cli.train -m optimizer.lr=1e-4,2e-4,5e-4 data.batch_size=128,256
"""

import os
from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    メイントレーニング関数

    Hydraが設定を自動的に読み込み、コマンドラインからのオーバーライドを適用します。

    Args:
        cfg: Hydraによって読み込まれた設定
    """

    # 設定を表示（デバッグ用）
    # print("=" * 80)
    # print("Training Configuration:")
    # print("=" * 80)
    # print(OmegaConf.to_yaml(cfg))
    # print("=" * 80)

    # シード設定
    L.seed_everything(cfg.seed, workers=True)

    # DataModule初期化
    print("\n[1/4] Initializing DataModule...")
    datamodule = hydra.utils.instantiate(cfg.data)
    print(f"  ✓ Data directory: {datamodule.data_dir}")
    print(f"  ✓ Global batch size: {datamodule.global_batch_size}")
    print(f"  ✓ Micro batch size: {datamodule.micro_batch_size}")

    # トレーニングスケジュールの計算
    # eval:インターポレーションはOmegaConfでサポートされないため、Pythonで計算
    print("\nCalculating training schedule...")
    model_params = cfg.model.num_parameters
    global_batch_size = datamodule.global_batch_size
    block_size = cfg.training.block_size
    flops_budget = cfg.training.flops_budget

    # max_steps = flops_budget / (6 * model_params * tokens_per_step)
    tokens_per_step = global_batch_size * block_size
    max_steps = int(flops_budget / (6 * model_params * tokens_per_step))
    warmup_steps = max(max_steps // 100, 100)

    # gradient_accumulation_steps = global_batch_size // (devices * micro_batch_size)
    devices = cfg.trainer.devices if isinstance(cfg.trainer.devices, int) else len(cfg.trainer.devices)
    gradient_accumulation_steps = global_batch_size // (devices * datamodule.micro_batch_size)

    print(f"  ✓ Max steps: {max_steps}")
    print(f"  ✓ Warmup steps: {warmup_steps}")
    print(f"  ✓ Gradient accumulation steps: {gradient_accumulation_steps}")

    # 設定を更新（後で参照するため）
    OmegaConf.update(cfg, "training.model_params", model_params)
    OmegaConf.update(cfg, "training.global_batch_size", global_batch_size)
    OmegaConf.update(cfg, "training.max_steps", max_steps)
    OmegaConf.update(cfg, "training.warmup_steps", warmup_steps)
    OmegaConf.update(cfg, "training.gradient_accumulation_steps", gradient_accumulation_steps)

    # Model初期化
    print("\n[2/4] Initializing Model...")
    # optimizer_config と scheduler_config を Model に渡す
    model_kwargs = {
        "model_config": cfg.model.model_config,
        "optimizer_config": OmegaConf.to_container(cfg.optimizer, resolve=True),
        "scheduler_config": OmegaConf.to_container(cfg.scheduler, resolve=True),
        "num_parameters": cfg.model.num_parameters,
    }
    model = hydra.utils.instantiate(cfg.model, **model_kwargs, _recursive_=False)
    print(f"  ✓ Model: {cfg.model.model_config.name}")
    print(f"  ✓ Parameters: {cfg.model.num_parameters / 1e6:.2f}M")
    print(f"  ✓ Layers: {cfg.model.model_config.n_layer}")
    print(f"  ✓ Hidden size: {cfg.model.model_config.n_embd}")

    # Callbacks初期化
    print("\n[3/4] Initializing Callbacks...")
    callbacks = []
    if "callbacks" in cfg and cfg.callbacks is not None:
        for callback_name, callback_cfg in cfg.callbacks.items():
            if callback_cfg is not None:
                cb = hydra.utils.instantiate(callback_cfg)
                callbacks.append(cb)
                print(f"  ✓ {callback_name}: {type(cb).__name__}")

    # Logger初期化
    print("\n[4/4] Initializing Logger...")
    logger = None
    if "logger" in cfg and cfg.logger is not None:
        logger = hydra.utils.instantiate(cfg.logger)
        print(f"  ✓ Logger: {type(logger).__name__}")
        if isinstance(logger, WandbLogger):
            print(f"  ✓ Project: {logger._project}")
            print(f"  ✓ Name: {logger._name}")
    else:
        # デフォルトでCSVLoggerを使用
        logger = CSVLogger(save_dir=cfg.output_dir, name="training_logs")
        print(f"  ✓ Logger: CSVLogger")

    # Trainer初期化
    print("\n" + "=" * 80)
    print("Initializing Trainer...")
    print("=" * 80)

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)

    trainer = L.Trainer(
        **trainer_kwargs,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=cfg.output_dir,
    )

    print(f"  ✓ Accelerator: {trainer.accelerator}")
    print(f"  ✓ Devices: {cfg.trainer.devices}")
    print(f"  ✓ Precision: {cfg.trainer.precision}")
    print(f"  ✓ Max steps: {cfg.trainer.max_steps}")
    print(f"  ✓ Gradient accumulation: {cfg.training.gradient_accumulation_steps}")

    # トレーニング情報の表示
    print("\n" + "=" * 80)
    print("Training Schedule:")
    print("=" * 80)
    print(f"  FLOPs budget: {cfg.training.flops_budget / 1e18:.2f}e18")
    print(f"  Max steps: {cfg.training.max_steps}")
    print(f"  Warmup steps: {cfg.training.warmup_steps}")
    print(f"  Global batch size: {cfg.training.global_batch_size}")
    print(f"  Tokens per step: {cfg.training.global_batch_size * cfg.training.block_size}")
    print("=" * 80)

    # チェックポイントからの再開
    ckpt_path = None
    if "resume_from" in cfg and cfg.resume_from is not None:
        ckpt_path = cfg.resume_from
        print(f"\n⚠ Resuming from checkpoint: {ckpt_path}")

    # トレーニング開始
    print("\n" + "=" * 80)
    print("🚀 Starting Training...")
    print("=" * 80)

    try:
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise

    # トレーニング完了
    print("\n" + "=" * 80)
    print("✅ Training completed!")
    print("=" * 80)

    # 結果サマリー
    if hasattr(trainer, "callback_metrics"):
        print("\nFinal metrics:")
        for key, value in trainer.callback_metrics.items():
            print(f"  {key}: {value}")

    # チェックポイントパスの表示
    if callbacks:
        for cb in callbacks:
            if isinstance(cb, ModelCheckpoint):
                print(f"\n💾 Best checkpoint: {cb.best_model_path}")
                print(f"💾 Last checkpoint: {cb.last_model_path}")


if __name__ == "__main__":
    main()
