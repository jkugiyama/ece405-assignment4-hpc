import argparse
from contextlib import nullcontext
import json
import pickle
import statistics
import timeit
from dataclasses import asdict, dataclass
from logging import INFO, basicConfig, getLogger
from pathlib import Path
from typing import Any

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

logger = getLogger(__name__)
basicConfig(level=INFO)


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_SIZE_CONFIGS = {
    # Preset sizes from the assignment family; adjust if your handout differs.
    "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7b": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

DEFAULT_VOCAB_SIZE = 50257
DEFAULT_BATCH_SIZE = 8
DEFAULT_CONTEXT_LENGTH = 256


@dataclass(frozen=True)
class BenchmarkConfig:
    model: ModelConfig
    vocab_size: int
    batch_size: int
    context_length: int
    warmup_steps: int
    benchmark_steps: int
    run_backward: bool
    device: str
    amp_dtype: torch.dtype | None


def maybe_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def safe_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def make_batch(
    *,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device,
    )
    targets = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device,
    )
    return inputs, targets


def run_benchmark(cfg: BenchmarkConfig) -> dict[str, Any]:
    model_cfg = cfg.model

    model = BasicsTransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=model_cfg.d_model,
        d_ff=model_cfg.d_ff,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        rope_theta=10000.0,
    ).to(cfg.device)
    model.train()

    optimizer = AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp_dtype == torch.float16)

    device_type = cfg.device.split(":", maxsplit=1)[0]

    forward_times: list[float] = []
    backward_times: list[float] = []
    loss_times: list[float] = []

    logger.info(
        "Starting benchmark on %s | warmup=%d steps=%d mode=%s amp=%s",
        cfg.device,
        cfg.warmup_steps,
        cfg.benchmark_steps,
        "fwd+bwd" if cfg.run_backward else "fwd-only",
        "none" if cfg.amp_dtype is None else str(cfg.amp_dtype).replace("torch.", ""),
    )

    for step in range(cfg.warmup_steps + cfg.benchmark_steps):
        inputs, targets = make_batch(
            batch_size=cfg.batch_size,
            context_length=cfg.context_length,
            vocab_size=cfg.vocab_size,
            device=cfg.device,
        )
        optimizer.zero_grad(set_to_none=True)

        t1 = timeit.default_timer()

        amp_ctx = (
            torch.autocast(device_type=device_type, dtype=cfg.amp_dtype)
            if cfg.amp_dtype is not None
            else nullcontext()
        )
        with amp_ctx:
            outputs = model(inputs)

        maybe_sync(cfg.device)
        t2 = timeit.default_timer()

        loss = cross_entropy(outputs, targets)
        maybe_sync(cfg.device)
        t3 = timeit.default_timer()

        if cfg.run_backward:
            if cfg.amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            maybe_sync(cfg.device)

        t4 = timeit.default_timer()
        maybe_sync(cfg.device)

        if step < cfg.warmup_steps:
            continue

        forward_time = t2 - t1
        loss_time = t3 - t2
        backward_time = t4 - t3

        forward_times.append(forward_time)
        loss_times.append(loss_time)
        if cfg.run_backward:
            backward_times.append(backward_time)

        logical_step = step - cfg.warmup_steps + 1
        logger.info("Step %d", logical_step)
        logger.info("Forward:  %.6f sec", forward_time)
        logger.info("Loss:     %.6f sec", loss_time)
        if cfg.run_backward:
            logger.info("Backward: %.6f sec", backward_time)

    results: dict[str, Any] = {
        "model": asdict(model_cfg),
        "device": cfg.device,
        "run_backward": cfg.run_backward,
        "amp_dtype": None if cfg.amp_dtype is None else str(cfg.amp_dtype).replace("torch.", ""),
        "warmup_steps": cfg.warmup_steps,
        "benchmark_steps": cfg.benchmark_steps,
        "forward_mean_sec": statistics.mean(forward_times),
        "forward_std_sec": safe_stdev(forward_times),
        "loss_mean_sec": statistics.mean(loss_times),
        "loss_std_sec": safe_stdev(loss_times),
    }
    if cfg.run_backward:
        results["backward_mean_sec"] = statistics.mean(backward_times)
        results["backward_std_sec"] = safe_stdev(backward_times)

    # Capture peak CUDA memory (bytes) across the entire benchmark run.
    if cfg.device.startswith("cuda"):
        results["peak_memory_bytes"] = torch.cuda.max_memory_allocated(cfg.device)
        results["peak_memory_mb"] = results["peak_memory_bytes"] / (1024 ** 2)

    logger.info("RESULTS")
    logger.info("Forward Mean: %.6f sec", results["forward_mean_sec"])
    logger.info("Forward Std:  %.6f sec", results["forward_std_sec"])
    logger.info("Loss Mean:    %.6f sec", results["loss_mean_sec"])
    logger.info("Loss Std:     %.6f sec", results["loss_std_sec"])
    if cfg.run_backward:
        logger.info("Backward Mean: %.6f sec", results["backward_mean_sec"])
        logger.info("Backward Std:  %.6f sec", results["backward_std_sec"])
    if "peak_memory_mb" in results:
        logger.info("Peak CUDA Memory: %.2f MB", results["peak_memory_mb"])

    return results


def run_memory_profile(cfg: BenchmarkConfig, output_path: str) -> None:
    """Run a single forward (and optionally backward+optimizer) step under
    PyTorch's memory-history recorder and save a snapshot pickle for
    visualisation with pytorch.org/memory_viz."""

    if not cfg.device.startswith("cuda"):
        raise RuntimeError("Memory profiling requires a CUDA device.")

    model_cfg = cfg.model

    model = BasicsTransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=model_cfg.d_model,
        d_ff=model_cfg.d_ff,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        rope_theta=10000.0,
    ).to(cfg.device)
    model.train()

    optimizer = AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp_dtype == torch.float16)
    device_type = cfg.device.split(":", maxsplit=1)[0]

    inputs, targets = make_batch(
        batch_size=cfg.batch_size,
        context_length=cfg.context_length,
        vocab_size=cfg.vocab_size,
        device=cfg.device,
    )

    # Warm-up pass (not recorded) to avoid capturing one-time CUDA allocations.
    amp_ctx = (
        torch.autocast(device_type=device_type, dtype=cfg.amp_dtype)
        if cfg.amp_dtype is not None
        else nullcontext()
    )
    with amp_ctx:
        warmup_out = model(inputs)
    warmup_loss = cross_entropy(warmup_out, targets)
    if cfg.run_backward:
        if cfg.amp_dtype == torch.float16:
            scaler.scale(warmup_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            warmup_loss.backward()
            optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Start recording memory history.
    torch.cuda.memory._record_memory_history(max_entries=100000)

    with amp_ctx:
        outputs = model(inputs)
    torch.cuda.synchronize()

    loss = cross_entropy(outputs, targets)
    torch.cuda.synchronize()

    if cfg.run_backward:
        if cfg.amp_dtype == torch.float16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

    snapshot = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)  # stop recording

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(snapshot, f)

    logger.info("Memory snapshot saved to %s", out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark BasicsTransformerLM")
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=sorted(MODEL_SIZE_CONFIGS.keys()),
        help="Named model-size preset.",
    )
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--benchmark-steps", type=int, default=10)
    parser.add_argument(
        "--mode",
        type=str,
        default="fwd-bwd",
        choices=["fwd", "fwd-bwd"],
        help="Benchmark forward only or forward+backward.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device string (e.g., cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="fp16",
        choices=["none", "fp16", "bf16"],
        help="Automatic mixed precision dtype to use on CUDA.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON results.",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Run a single step under the PyTorch memory profiler instead of benchmarking.",
    )
    parser.add_argument(
        "--memory-output",
        type=str,
        default="memory_snapshot.pickle",
        help="Output path for the memory snapshot pickle file (used with --profile-memory).",
    )
    return parser.parse_args()


def resolve_model_config(args: argparse.Namespace) -> ModelConfig:
    base = MODEL_SIZE_CONFIGS[args.model_size]
    return ModelConfig(
        d_model=args.d_model if args.d_model is not None else base.d_model,
        d_ff=args.d_ff if args.d_ff is not None else base.d_ff,
        num_layers=args.num_layers if args.num_layers is not None else base.num_layers,
        num_heads=args.num_heads if args.num_heads is not None else base.num_heads,
    )


def main() -> None:
    args = parse_args()
    model = resolve_model_config(args)

    if model.d_model % model.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    if args.amp_dtype == "none":
        amp_dtype = None
    elif args.amp_dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif args.device.startswith("cuda"):
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    cfg = BenchmarkConfig(
        model=model,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        context_length=args.context_length,
        warmup_steps=args.warmup_steps,
        benchmark_steps=args.benchmark_steps,
        run_backward=args.mode == "fwd-bwd",
        device=args.device,
        amp_dtype=amp_dtype,
    )

    if args.profile_memory:
        run_memory_profile(cfg, args.memory_output)
    else:
        results = run_benchmark(cfg)
        if args.json:
            print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()