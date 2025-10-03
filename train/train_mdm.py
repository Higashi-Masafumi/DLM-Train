from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import lightning as L
import torch
from functools import partial
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader

from flash_attn.losses.cross_entropy import CrossEntropyLoss

from models.config import Config
from models.diffusionlm import Block, TransEncoder
from loader.packed_dataset import CombinedDataset, PackedDataset
from utils.speed_monitor import estimate_flops
from utils.utils import get_default_supported_precision, num_parameters, step_csv_logger

MODEL_PARAM_CONFIG = {
    "6": 6.294784,
    "19": 18.880896,
    "34": 33.563136,
    "48": 47.786688,
    "66": 65.54944,
    "75": 75.38752,
    "85": 85.21408,
    "113": 113.265408,
    "142": 141.581568,
    "170": 169.897728,
    "180": 179.856768,
    "206": 205.550464,
    "231": 231.24416,
    "268": 268.469248,
    "302": 302.027776,
    "336": 335.586304,
    "472": 471.90656,
    "551": 550.55744,
    "571": 571.001728,
    "629": 629.20832,
    "666": 666.168448,
    "717": 717.285888,
    "761": 761.335168,
    "831": 830.541312,
    "944": 943.796736,
    "1028": 1027.677952,
    "1233": 1233.213184,
    "1476": 1476.487168,
    "1678": 1677.826048,
    "2121": 2121.39328,
}

TRAIN_DATA_CONFIG: Sequence[Tuple[str, float]] = [("train_slim", 1.0), ("train_star", 0.0)]
VAL_DATA_CONFIG: Sequence[Tuple[str, float]] = [("validation", 1.0)]
DEFAULT_TRAIN_DATA_DIR = Path("/dataset/slim_star_combined")
DEFAULT_OUTPUT_ROOT = Path("workdir/scaling_debug")
VALIDATION_MC_SAMPLES = 32


@dataclass
class CLIArgs:
    model: int
    flops: float
    batch_size: int
    nodes_num: int
    devices: Optional[int]
    train_data_dir: Path
    val_data_dir: Optional[Path]
    precision: Optional[str]
    seed: int
    resume_from: Optional[Path]
    output_dir: Path
    train_chunks: int
    val_chunks: int
    log_interval: int
    eval_interval: Optional[int]
    save_interval: int
    learning_rate: float
    min_lr: float
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    decay_lr: bool
    max_step_override: Optional[int]


@dataclass
class TrainingSchedule:
    model_name: str
    model_param_scale: float
    global_batch_size: int
    per_device_batch: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    min_lr: float
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    decay_lr: bool
    warmup_iters: int
    max_step: int
    max_iters: int
    lr_decay_iters: int
    log_step_interval: int
    eval_step_interval: Optional[int]
    save_step_interval: int
    eval_iters: int
    seed: int
    output_dir: Path
    train_chunks: int
    val_chunks: int


def parse_args() -> CLIArgs:
    parser = argparse.ArgumentParser(description="Train DiffusionLM with Fabric")
    parser.add_argument("--model", type=int, required=True, help="Diff_LLaMA size in millions, e.g. 170")
    parser.add_argument("--flops", type=float, required=True, help="Target training compute in exaFLOPs (e18)")
    parser.add_argument("--batch-size", type=int, default=256, help="Global batch size per optimizer step")
    parser.add_argument("--nodes-num", type=int, default=1, help="Number of nodes participating in training")
    parser.add_argument("--devices", type=int, default=None, help="Requested CUDA devices (default: detected)")
    parser.add_argument("--train-data-dir", type=Path, default=DEFAULT_TRAIN_DATA_DIR, help="Directory with packed train_*.bin files")
    parser.add_argument("--val-data-dir", type=Path, default=None, help="Directory with packed validation_*.bin files")
    parser.add_argument("--precision", type=str, default=None, help="Fabric precision override, e.g. bf16")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument("--resume-from", type=Path, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory for checkpoints and logs")
    parser.add_argument("--train-chunks", type=int, default=8, help="Number of PackedDataset chunks to buffer during training")
    parser.add_argument("--val-chunks", type=int, default=1, help="Number of PackedDataset chunks to buffer during validation")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N optimizer steps")
    parser.add_argument("--eval-interval", type=int, default=0, help="Validate every N optimizer steps (0 disables validation)")
    parser.add_argument("--save-interval", type=int, default=5000, help="Save a checkpoint every N optimizer steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-lr-decay", action="store_true", help="Disable cosine LR decay")
    parser.add_argument("--max-step", type=int, default=None, help="Override maximum optimizer steps")
    args = parser.parse_args()

    eval_interval = None if args.eval_interval <= 0 else args.eval_interval

    return CLIArgs(
        model=args.model,
        flops=args.flops,
        batch_size=args.batch_size,
        nodes_num=args.nodes_num,
        devices=args.devices,
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        precision=args.precision,
        seed=args.seed,
        resume_from=args.resume_from,
        output_dir=args.output_dir,
        train_chunks=args.train_chunks,
        val_chunks=args.val_chunks,
        log_interval=max(1, args.log_interval),
        eval_interval=eval_interval,
        save_interval=max(1, args.save_interval),
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        grad_clip=args.grad_clip,
        decay_lr=not args.no_lr_decay,
        max_step_override=args.max_step,
    )


def detect_cuda_device_count() -> int:
    try:
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    return 0


def resolve_device_count(requested: Optional[int], available: int) -> int:
    if available == 0:
        return 1
    if requested is None:
        return available
    if requested <= 0:
        return 1
    if requested > available:
        print(f"Requested {requested} devices but only {available} available; using {available}.")
        return available
    return requested


def suggest_micro_batch(model_size_millions: int) -> int:
    if model_size_millions <= 20:
        return 32
    if model_size_millions <= 50:
        return 16
    if model_size_millions <= 1000:
        return 8
    return 4


def derive_schedule(args: CLIArgs, model_config: Config, device_count: int) -> TrainingSchedule:
    model_key = str(args.model)
    if model_key not in MODEL_PARAM_CONFIG:
        raise ValueError(f"Unknown model size {args.model}. Available keys: {', '.join(sorted(MODEL_PARAM_CONFIG))}")
    model_name = f"Diff_LLaMA_{args.model}M"
    global_target = max(1, args.batch_size // max(1, args.nodes_num))
    per_device_target = max(1, global_target // max(1, device_count))

    suggested_micro = suggest_micro_batch(args.model)
    micro_batch_size = min(suggested_micro, per_device_target)
    gradient_accumulation_steps = math.ceil(per_device_target / micro_batch_size)
    per_device_batch = micro_batch_size * gradient_accumulation_steps
    global_batch_size = per_device_batch * device_count
    if global_batch_size != global_target:
        print(
            f"Adjusted global batch size from {global_target} to {global_batch_size} "
            f"to satisfy micro-batch scheduling."
        )

    model_param_scale = MODEL_PARAM_CONFIG[model_key]
    if args.max_step_override is not None:
        max_step = max(1, args.max_step_override)
    else:
        denom = 6 * model_param_scale * global_batch_size * model_config.block_size * max(1, args.nodes_num)
        if denom == 0:
            raise ValueError("Invalid schedule: denominator for max_step computation is zero")
        max_step = max(1, int(args.flops * 1e12 / denom))

    warmup_iters = max(100, max_step // 100)
    max_iters = max_step * gradient_accumulation_steps
    lr_decay_iters = max_iters
    eval_iters = max(1, (100 * 1024) // max(1, global_batch_size))
    run_dir = args.output_dir / f"mdm-{args.model}M-{args.flops}e18"

    return TrainingSchedule(
        model_name=model_name,
        model_param_scale=model_param_scale,
        global_batch_size=global_batch_size,
        per_device_batch=per_device_batch,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        grad_clip=args.grad_clip,
        decay_lr=args.decay_lr,
        warmup_iters=warmup_iters,
        max_step=max_step,
        max_iters=max_iters,
        lr_decay_iters=lr_decay_iters,
        log_step_interval=args.log_interval,
        eval_step_interval=args.eval_interval,
        save_step_interval=args.save_interval,
        eval_iters=eval_iters,
        seed=args.seed,
        output_dir=run_dir,
        train_chunks=max(1, args.train_chunks),
        val_chunks=max(1, args.val_chunks),
    )


def learning_rate_for_iteration(iteration: int, schedule: TrainingSchedule) -> float:
    if not schedule.decay_lr:
        return schedule.learning_rate
    if iteration < schedule.warmup_iters:
        return schedule.learning_rate * iteration / max(1, schedule.warmup_iters)
    if iteration >= schedule.lr_decay_iters:
        return schedule.min_lr
    decay_ratio = (iteration - schedule.warmup_iters) / max(1, schedule.lr_decay_iters - schedule.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return schedule.min_lr + coeff * (schedule.learning_rate - schedule.min_lr)


def normalize_weights(config: Sequence[Tuple[str, float]]) -> Sequence[float]:
    weights = [weight for _, weight in config]
    total = sum(weights)
    if total == 0:
        count = len(weights)
        return [1.0 / count] * count if count else []
    return [weight / total for weight in weights]


def gather_filenames(data_dir: Path, prefix: str) -> Sequence[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")
    files = sorted(data_dir.glob(f"{prefix}*"))
    if not files:
        raise FileNotFoundError(f"No files matching '{prefix}*' found in {data_dir}")
    return files


def make_packed_dataset(
    filenames: Sequence[Path],
    block_size: int,
    n_chunks: int,
    seed: int,
    shuffle: bool,
    fabric: L.Fabric,
) -> PackedDataset:
    chunk_budget = max(1, min(n_chunks, len(filenames)))
    file_list = [str(path) for path in filenames]
    return PackedDataset(
        filenames=file_list,
        n_chunks=chunk_budget,
        block_size=block_size,
        seed=seed,
        shuffle=shuffle,
        wrap=True,
        num_processes=fabric.world_size,
        process_rank=fabric.global_rank,
    )


def build_dataloader(
    fabric: L.Fabric,
    data_dir: Path,
    layout: Sequence[Tuple[str, float]],
    block_size: int,
    batch_size: int,
    seed: int,
    shuffle: bool,
    chunks: int,
    pin_memory: bool,
) -> DataLoader:
    datasets = []
    weights = normalize_weights(layout)
    for idx, (prefix, _) in enumerate(layout):
        files = list(gather_filenames(data_dir, prefix))
        rng = random.Random(seed)
        rng.shuffle(files)
        dataset_seed = seed + fabric.global_rank
        dataset = make_packed_dataset(
            filenames=files,
            block_size=block_size,
            n_chunks=chunks,
            seed=dataset_seed,
            shuffle=shuffle,
            fabric=fabric,
        )
        datasets.append(dataset)
    combined = CombinedDataset(datasets=datasets, seed=seed, weights=weights)
    return DataLoader(combined, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)


def forward_process(batch: torch.Tensor, total_dim: int = 32000, eps: float = 1e-3):
    b, l = batch.shape
    t = torch.rand((b,), device=batch.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    mask_indices = torch.rand((b, l), device=batch.device) < p_mask
    noisy_batch = torch.where(mask_indices, total_dim, batch)
    return noisy_batch, mask_indices, p_mask


def validate(
    fabric: L.Fabric,
    model: torch.nn.Module,
    dataloader: Optional[DataLoader],
    schedule: TrainingSchedule,
) -> Optional[torch.Tensor]:
    if dataloader is None:
        return None

    criterion = CrossEntropyLoss(reduction="none")
    model.eval()
    losses = []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step >= schedule.eval_iters:
                break
            input_ids = batch[:, : model.config.block_size].contiguous()
            mc_losses = torch.zeros(VALIDATION_MC_SAMPLES, device=fabric.device)
            for i in range(VALIDATION_MC_SAMPLES):
                noisy_input, mask_indices, p_mask = forward_process(input_ids)
                with fabric.autocast():
                    logits = model(noisy_input)
                loss = criterion(logits[mask_indices], input_ids[mask_indices]) / p_mask[mask_indices]
                mc_losses[i] = loss.sum() / (input_ids.size(0) * input_ids.size(1))
            losses.append(mc_losses.mean())
    if not losses:
        mean_loss = torch.tensor(0.0, device=fabric.device)
    else:
        mean_loss = torch.stack(losses).mean()
    model.train()
    return mean_loss


def save_checkpoint(fabric: L.Fabric, state: dict, schedule: TrainingSchedule) -> None:
    schedule.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = schedule.output_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
    fabric.save(ckpt_path, state)
    fabric.print(f"Saved checkpoint to {ckpt_path}")


def train(
    fabric: L.Fabric,
    model: TransEncoder,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    schedule: TrainingSchedule,
    estimated_flops: float,
    state: dict,
) -> None:
    criterion = CrossEntropyLoss(reduction="none")
    iter_num = state.get("iter_num", 0)
    step_count = state.get("step_count", 0)

    start_time = time.perf_counter()
    fabric.print(
        f"Starting training at iter {iter_num}, step {step_count}. "
        f"Estimated TFLOPs (per device): {estimated_flops / 1e12:.2f}"
    )

    for batch in train_loader:
        if iter_num >= schedule.max_iters:
            break

        lr = learning_rate_for_iteration(iter_num, schedule)
        for group in optimizer.param_groups:
            group["lr"] = lr

        input_ids = batch[:, : model.config.block_size].contiguous()
        noisy_input, mask_indices, p_mask = forward_process(input_ids)

        is_accumulating = ((iter_num + 1) % schedule.gradient_accumulation_steps) != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            with fabric.autocast():
                logits = model(noisy_input)
            loss = criterion(logits[mask_indices], input_ids[mask_indices]) / p_mask[mask_indices]
            loss = loss.sum() / (input_ids.size(0) * input_ids.size(1))
            fabric.backward(loss / schedule.gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=schedule.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            if schedule.eval_step_interval and (step_count % schedule.eval_step_interval == 0):
                val_loss = validate(fabric, model, val_loader, schedule)
                if val_loss is not None:
                    fabric.print(f"[step {step_count}] val_loss={val_loss.item():.4f}")

            if schedule.save_step_interval and (step_count % schedule.save_step_interval == 0):
                save_checkpoint(fabric, state, schedule)

            if schedule.log_step_interval and (step_count % schedule.log_step_interval == 0):
                elapsed = time.perf_counter() - start_time
                tokens_processed = step_count * schedule.global_batch_size * model.config.block_size
                fabric.print(
                    f"[step {step_count}] iter={iter_num} loss={loss.item():.4f} lr={lr:.2e} "
                    f"tokens={tokens_processed:,} elapsed={elapsed / 3600:.2f}h"
                )

        iter_num += 1
        state["iter_num"] = iter_num
        state["step_count"] = step_count

        if iter_num >= schedule.max_iters:
            break

    fabric.print(f"Finished training at iter {iter_num}, step {step_count}.")


def main() -> None:
    args = parse_args()
    available_cuda = detect_cuda_device_count()
    device_count = resolve_device_count(args.devices, available_cuda)

    config = Config.from_name(f"Diff_LLaMA_{args.model}M")
    schedule = derive_schedule(args, config, device_count)
    schedule.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Resolved devices={device_count}, global_batch={schedule.global_batch_size}, "
        f"micro_batch={schedule.micro_batch_size}, grad_acc={schedule.gradient_accumulation_steps}"
    )

    precision = args.precision or get_default_supported_precision(training=True, tpu=False)
    logger = step_csv_logger(str(schedule.output_dir), schedule.model_name, flush_logs_every_n_steps=schedule.log_step_interval)

    strategy: Optional[FSDPStrategy]
    if device_count > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy=None,
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"  # type: ignore[assignment]

    fabric = L.Fabric(devices=device_count, strategy=strategy, precision=precision, loggers=[logger])
    fabric.print(f"Using precision={precision}")
    fabric.seed_everything(schedule.seed)

    pin_memory = device_count > 0 and torch.cuda.is_available()
    train_loader = build_dataloader(
        fabric=fabric,
        data_dir=args.train_data_dir,
        layout=TRAIN_DATA_CONFIG,
        block_size=config.block_size + 1,
        batch_size=schedule.micro_batch_size,
        seed=schedule.seed,
        shuffle=True,
        chunks=schedule.train_chunks,
        pin_memory=pin_memory,
    )
    val_loader = None
    if args.val_data_dir is not None and args.eval_interval:
        val_loader = build_dataloader(
            fabric=fabric,
            data_dir=args.val_data_dir,
            layout=VAL_DATA_CONFIG,
            block_size=config.block_size + 1,
            batch_size=schedule.micro_batch_size,
            seed=schedule.seed,
            shuffle=False,
            chunks=schedule.val_chunks,
            pin_memory=pin_memory,
        )

    train_loader = fabric.setup_dataloaders(train_loader)
    if val_loader is not None:
        val_loader = fabric.setup_dataloaders(val_loader)

    fabric.print(f"Loading model with config: {config.model_dump()}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = TransEncoder(config)
        model.apply(partial(model._init_weights, n_layer=config.n_layer))
    fabric.print(f"Model instantiated in {time.perf_counter() - t0:.2f}s with {num_parameters(model):,} params")

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=schedule.learning_rate,
        weight_decay=schedule.weight_decay,
        betas=(schedule.beta1, schedule.beta2),
        foreach=False,
    )
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "iter_num": 0, "step_count": 0}

    if args.resume_from:
        fabric.print(f"Resuming from checkpoint {args.resume_from}")
        fabric.load(args.resume_from, state)

    with torch.device("meta"):
        meta_model = TransEncoder(config)
        estimated_flops = estimate_flops(meta_model) * schedule.micro_batch_size
    del meta_model

    train(
        fabric=fabric,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        schedule=schedule,
        estimated_flops=estimated_flops,
        state=state,
    )

    if fabric.device.type == "cuda":
        fabric.print(f"Peak memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
