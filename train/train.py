import argparse
import glob
import math
import random
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from flash_attn.losses.cross_entropy import CrossEntropyLoss

from loader.packed_dataset import CombinedDataset, PackedDataset

# Project-local imports
from models.diffusionlm import Block, Config, TransEncoder
from utils.speed_monitor import SpeedMonitorFabric as Monitor
from utils.speed_monitor import estimate_flops
from utils.utils import (  # noqa: F401 (kept for parity; not used by default)
    chunked_cross_entropy,
)
from utils.utils import get_default_supported_precision, num_parameters, step_csv_logger

# ======== CLI ========

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed args: model(M params), nodes_num, flops(*1e18), batch_size.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=int, required=True, help="Model size in millions of parameters.")
    parser.add_argument("--nodes_num", type=int, default=1, help="Number of nodes.")
    parser.add_argument("--flops", type=float, required=True, help="FLOPs budget, in units of *1e18.")
    parser.add_argument("--batch_size", type=int, default=256, help="Global batch size across all nodes.")
    return parser.parse_args()


# ======== Hyper-parameters & globals (initialized after args) ========

args = parse_args()
model_name: str = f"Diff_LLaMA_{args.model}M"
out_dir: Path = Path("workdir")

# Model size (key=millions as str) -> parameter count proxy (for FLOPs calc)
model_para_config: dict[str, float] = {
    "6": 6.294784, "19": 18.880896, "34": 33.563136, "48": 47.786688, "66": 65.54944,
    "85": 85.21408, "75": 75.38752, "113": 113.265408, "142": 141.581568, "170": 169.897728,
    "180": 179.856768, "206": 205.550464, "231": 231.24416, "268": 268.469248, "302": 302.027776,
    "336": 335.586304, "472": 471.90656, "551": 550.55744, "571": 571.001728, "629": 629.20832,
    "666": 666.168448, "717": 717.285888, "761": 761.335168, "831": 830.541312, "944": 943.796736,
    "1028": 1027.677952, "1233": 1233.213184, "1476": 1476.487168, "1678": 1677.826048, "2121": 2121.39328,
}

# Training scale setup
num_of_devices: int = 1
global_batch_size: int = int(args.batch_size / args.nodes_num)
learning_rate: float = 2e-4

# Memory-aware micro-batch
if args.model <= 20:
    micro_batch_size: int = 32
elif args.model <= 50:
    micro_batch_size = 16
elif args.model <= 1000:
    micro_batch_size = 8
else:
    micro_batch_size = 4

# Steps from FLOPs budget (rough scaling law style)
max_step: int = int(
    args.flops * 1e12 / (6 * model_para_config[f"{args.model}"] * global_batch_size * 2048) / args.nodes_num
)
warmup_steps: int = max(int(max_step / 100), 100)
log_step_interval: int = 10
eval_iters: int = int(100 * 1024 / global_batch_size)
save_step_interval: int = 5000
eval_step_interval: int = 999_999_999_999  # effectively disables eval during training by default

# Optimizer & training knobs
weight_decay: float = 1e-1
beta1: float = 0.9
beta2: float = 0.95
grad_clip: float = 1.0
decay_lr: bool = True
min_lr: float = 2e-5

# Per-device batch and gradient accumulation
batch_size: int = global_batch_size // num_of_devices
gradient_accumulation_steps: int = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0, "micro_batch_size is too large for per-device batch_size"

# Convert step-based to iteration-based (micro-batch) units
warmup_iters: int = warmup_steps * gradient_accumulation_steps
max_iters: int = max_step * gradient_accumulation_steps
lr_decay_iters: int = max_iters
log_iter_interval: int = log_step_interval * gradient_accumulation_steps

# Dataset mixture config (prefix, weight); star=0.0 means disabled by default
train_data_config: list[tuple[str, float]] = [
    ("train_slim", 1.0),
    ("train_star", 0.0),
]
val_data_config: list[tuple[str, float]] = [
    ("validation", 1.0),
]

# Collect simple hparams for logging
hparams: dict[str, int | float | str] = {
    k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")
}
logger = step_csv_logger("out", model_name, flush_logs_every_n_steps=log_iter_interval)


# ======== Data structures ========

@dataclass
class TrainState:
    """Container for training state to avoid untyped dict access."""
    model: torch.nn.Module
    optimizer: Optimizer
    hparams: dict[str, int | float | str]
    iter_num: int = 0      # micro-iteration counter (pre-accumulation)
    step_count: int = 0    # optimizer steps (post-accumulation)


# ======== Core utilities ========

def forward_process(
    batch: torch.Tensor,
    total_dim: int = 32000,
    eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply random token masking (diffusion/MAE-style) on input IDs.

    Args:
        batch: Input token IDs of shape (B, L), dtype=int64 expected.
        total_dim: Special token ID used as mask (typically vocab_size or vocab_size+offset).
        eps: Minimum masking probability per sample (avoids 0/1 extremes).

    Returns:
        noisy_batch: Masked input token IDs, same shape as `batch`.
        mask_indices: Boolean mask (B, L) of where tokens were replaced.
        p_mask: Per-token masking probability (B, L) that was applied.
    """
    b, l = batch.shape
    t = torch.rand((b,), device=batch.device)
    p_mask = (1.0 - eps) * t + eps                 # (B,)
    p_mask = p_mask[:, None].repeat(1, l)          # (B, L)
    mask_indices = torch.rand((b, l), device=batch.device) < p_mask
    noisy_batch = torch.where(mask_indices, torch.tensor(total_dim, device=batch.device), batch)
    return noisy_batch, mask_indices, p_mask


def setup(
    devices: int = 1,
    train_data_dir: Path = Path("/dataset/slim_star_combined"),
    val_data_dir: Path = Path("/dataset/slim_star_combined"),
    precision: str | None = None,
    tpu: bool = False,
    resume: bool | Path = True,
) -> None:
    """Initialize Fabric, loggers, strategy and start training.

    Args:
        devices: Number of local devices (GPUs/TPU cores). For TPU multi-host, Fabric uses per-host count.
        train_data_dir: Directory containing training shards (prefix-based).
        val_data_dir: Directory containing validation shards (prefix-based).
        precision: Fabric precision string (e.g. 'bf16-mixed'). If None, choose a default.
        tpu: Whether to run on TPU/XLA.
        resume: True to auto-pick latest checkpoint, False to start fresh, or a Path to a specific checkpoint.
    """
    global out_dir
    hp_name = f"mdm-{args.model}M-{args.flops}"
    out_dir = Path("workdir/scaling_debug") / hp_name

    wandb_logger = WandbLogger(name=f"{hp_name}-mc", save_dir=out_dir, project="scaling")
    chosen_precision = precision or get_default_supported_precision(training=True, tpu=tpu)

    if devices > 1:
        if tpu:
            devices = "auto"  # let Fabric detect per-host device count
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy=None,
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
            )
    else:
        strategy = "auto"

    fabric = L.Fabric(
        devices=devices,
        strategy=strategy,
        precision=chosen_precision,
        loggers=[logger, wandb_logger],
    )
    fabric.print(hparams)

    main(fabric, train_data_dir, val_data_dir, resume)


def main(
    fabric: L.Fabric,
    train_data_dir: Path,
    val_data_dir: Path,
    resume: bool | Path,
) -> None:
    """Build model/dataloaders, (optionally) resume, and run training."""
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)

    # Dataloaders (IterableDatasets)
    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=3407,
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(3407)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = TransEncoder(config)
        # Custom init depending on n_layer
        model.apply(partial(model._init_weights, n_layer=config.n_layer))
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        foreach=False,
    )
    optimizer = fabric.setup_optimizers(optimizer)

    state = TrainState(model=model, optimizer=optimizer, hparams=hparams)

    # Auto-resume
    if resume is True:
        import re

        def extract_number(filename: Path) -> int:
            m = re.search(r"iter-(\d+)-ckpt\.pth", str(filename))
            return int(m.group(1)) if m else 0

        try:
            resume = sorted(out_dir.glob("*.pth"), key=extract_number)[-1]
        except Exception:
            resume = False

    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state.__dict__)  # Fabric expects a mapping

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, monitor, resume=bool(resume))
    fabric.print(f"Training time: {(time.perf_counter() - train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(
    fabric: L.Fabric,
    state: TrainState,
    train_dataloader: Iterable[torch.Tensor],
    val_dataloader: Iterable[torch.Tensor] | None,
    monitor: Monitor,
    resume: bool,
) -> None:
    """Run the training loop with gradient accumulation and (optional) eval/ckpt.

    Args:
        fabric: Fabric runtime.
        state: Mutable training state.
        train_dataloader: Iterable of (B, L+1) token ID tensors.
        val_dataloader: Optional iterable for evaluation.
        monitor: Speed/TFLOPs monitor.
        resume: Whether we're resuming and need to fast-forward the dataloader.
    """
    model = state.model
    optimizer = state.optimizer

    # FLOPs estimate (meta device)
    with torch.device("meta"):
        meta_model = TransEncoder(model.config)
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        del meta_model

    total_lengths: int = 0
    total_t0 = time.perf_counter()

    initial_iter = state.iter_num
    curr_iter = 0

    loss_func = CrossEntropyLoss(reduction="none")

    for train_data in train_dataloader:
        # Fast-forward the iterable to align with saved iter_num (simple but effective)
        if resume:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                fabric.print(f"resume finished, taken {time.perf_counter() - total_t0:.2f} seconds")

        if state.iter_num >= max_iters:
            break

        # LR schedule
        lr = get_lr(state.iter_num) if decay_lr else learning_rate
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        iter_t0 = time.perf_counter()

        # Use exactly block_size tokens (drop the +1 target position from create_dataloaders)
        input_ids = train_data[:, 0: model.config.block_size].contiguous()

        # Diffusion-style masking
        noisy_input, mask_indices, p_mask = forward_process(input_ids)

        is_accumulating = (state.iter_num + 1) % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(noisy_input)
            # Masked-token CE with 1/p normalization to keep expectation stable
            loss_vec = loss_func(logits[mask_indices], input_ids[mask_indices]) / p_mask[mask_indices]
            loss = loss_vec.sum() / (input_ids.shape[0] * input_ids.shape[1])

            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state.step_count += 1

        state.iter_num += 1
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()

        fabric.print(
            f"iter {state.iter_num} step {state.step_count}: "
            f"loss {loss.item():.4f}, iter time {(t1 - iter_t0) * 1000:.2f}ms"
            f"{' (optimizer.step)' if not is_accumulating else ''} "
            f"remaining time: {(t1 - total_t0) / (state.iter_num - initial_iter) * (max_iters - state.iter_num) / 3600:.2f} h "
            f"({(t1 - total_t0) / (state.iter_num - initial_iter) * (max_iters - state.iter_num) / 3600 / 24:.2f} d)"
        )

        monitor.on_train_batch_end(
            state.iter_num * micro_batch_size,
            t1 - total_t0,
            fabric.world_size,
            state.step_count,
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss=loss.item(),
        )

        # Validation (disabled by default via huge interval)
        should_eval = (
            val_dataloader is not None
            and not is_accumulating
            and (state.step_count % eval_step_interval == 0 or state.step_count == max_step)
        )
        if should_eval and val_dataloader is not None:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader)
            dt = time.perf_counter() - t0
            monitor.eval_end(dt)
            fabric.print(f"step {state.iter_num}: val loss {val_loss:.4f}, val time: {dt * 1000:.2f}ms")
            total_tokens = model.config.block_size * (state.iter_num + 1) * micro_batch_size * fabric.world_size
            fabric.log_dict({"metric/val_loss": float(val_loss.item()), "total_tokens": total_tokens}, state.step_count)
            fabric.log_dict({"metric/val_ppl": math.exp(float(val_loss.item())), "total_tokens": total_tokens}, state.step_count)
            fabric.barrier()

        # Checkpointing
        should_save = not is_accumulating and (state.step_count % save_step_interval == 0 or state.step_count == max_step)
        if should_save:
            checkpoint_path = out_dir / f"iter-{state.iter_num:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            # Fabric expects a mapping; use state.__dict__ to serialize dataclass
            fabric.save(checkpoint_path, state.__dict__)


@torch.no_grad()
def validate(
    fabric: L.Fabric,
    model: torch.nn.Module,
    val_dataloader: Iterable[torch.Tensor],
) -> torch.Tensor:
    """Evaluate with Monte-Carlo masking to reduce variance.

    Runs over up to `eval_iters` batches; for each batch, re-samples masking 128 times.

    Returns:
        torch.Tensor: Mean validation loss across processes (scalar tensor).
    """
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break

        mc_num = 128
        mc_loss = torch.zeros(mc_num, device=fabric.device)
        for i in range(mc_num):
            input_ids = val_data[:, 0: model.config.block_size].contiguous()
            noisy_input, mask_indices, p_mask = forward_process(input_ids)
            logits = model(noisy_input)
            loss = torch.nn.functional.cross_entropy(
                logits[mask_indices], input_ids[mask_indices], reduction="none"
            ) / p_mask[mask_indices]
            mc_loss[i] = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

        losses[k] = mc_loss.mean().item()

    # Mean across ranks then across iterations
    losses = fabric.all_reduce(losses, reduce_op="mean")
    out = losses.mean()
    model.train()
    return out


def create_dataloader(
    batch_size: int,
    block_size: int,
    data_dir: Path,
    fabric: L.Fabric,
    shuffle: bool = True,
    seed: int = 12345,
    split: str = "train",
) -> DataLoader:
    """Create a DataLoader over a CombinedDataset of shard prefixes.

    The underlying datasets are Iterable; `n_chunks` controls buffer size
    (also impacts effective shuffle).

    Args:
        batch_size: Per-device batch size.
        block_size: Sequence length (+1 applied by the caller if needed).
        data_dir: Directory containing files like f\"{prefix}*\".
        fabric: Fabric instance to set world size and rank for sharding.
        shuffle: Shuffle file list (not token-level) for variability.
        seed: Base RNG seed; rank offset is added internally.
        split: 'train' or 'validation' to select `*_data_config`.

    Returns:
        DataLoader: Non-shuffling torch DataLoader over a CombinedDataset.
    """
    datasets: list[torch.utils.data.IterableDataset] = []
    data_config = train_data_config if split == "train" else val_data_config

    for prefix, _ in data_config:
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        random.seed(seed)
        random.shuffle(filenames)

        dataset = PackedDataset(
            filenames=filenames,
            n_chunks=8 if split == "train" else 1,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed + fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Ensure dataset preparation has been run."
        )

    weights = [w for _, w in data_config]
    sum_weights = sum(weights) if sum(weights) > 0 else 1.0
    norm_weights = [w / sum_weights for w in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=norm_weights)
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric: L.Fabric,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Path | None = None,
    seed: int = 12345,
) -> tuple[DataLoader, DataLoader | None]:
    """Create train/val DataLoaders with `block_size + 1` effective length.

    Returns:
        (train_loader, val_loader_or_None)
    """
    effective_block = block_size + 1
    train_loader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train",
    )
    val_loader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block,
            fabric=fabric,
            data_dir=val_data_dir if val_data_dir is not None else Path("."),
            shuffle=False,
            seed=seed,
            split="validation",
        )
        if val_data_dir is not None
        else None
    )
    return train_loader, val_loader


def get_lr(it: int) -> float:
    """Cosine LR with linear warmup in *iteration* units (micro-batches).

    Args:
        it: Current micro-iteration (pre-accumulation).

    Returns:
        float: Learning rate for this iteration.
    """
    if it < warmup_iters:
        return learning_rate * it / max(1, warmup_iters)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, (lr_decay_iters - warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # in [0, 1]
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # If you see: "Expected is_sm80 to be true, but got false", uncomment below:
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    setup()
