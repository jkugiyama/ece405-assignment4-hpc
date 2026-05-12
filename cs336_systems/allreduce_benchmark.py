"""
All-reduce operation benchmark script for distributed training.

Benchmarks torch.distributed.all_reduce() with different configurations:
- Backends: Gloo (CPU), NCCL (GPU)
- Data sizes: 1MB, 10MB, 100MB, 1GB
- Number of processes: 2, 4, 6
"""

import json
import logging
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@dataclass
class BenchConfig:
    """Configuration for an all-reduce benchmark."""

    backend: str
    device_type: str
    data_size_mb: float
    world_size: int
    warmup_iterations: int = 5
    benchmark_iterations: int = 10


@dataclass
class BenchResult:
    """Result from an all-reduce benchmark."""

    backend: str
    device_type: str
    data_size_mb: float
    world_size: int
    rank: int
    min_ms: float
    max_ms: float
    mean_ms: float
    stdev_ms: float


def _get_device(rank: int, device_type: str) -> torch.device:
    """Get the appropriate device for a given rank."""
    if device_type == "cpu":
        return torch.device("cpu")
    elif device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        return torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    else:
        raise ValueError(f"Unknown device type: {device_type}")


def _setup_distributed(rank: int, world_size: int, backend: str, timeout_seconds: int = 300) -> str:
    """
    Set up the distributed process group.

    Returns the device to use for this rank.
    """
    # Set up environment variables
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    # Determine device based on backend
    if backend == "gloo":
        device_type = "cpu"
    elif backend == "nccl":
        device_type = "cuda"
    else:
        raise ValueError(f"Unknown backend: {backend}")

    device = _get_device(rank, device_type)

    # Initialize the process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=torch.distributed.timedelta(seconds=timeout_seconds),
    )

    return device


def _cleanup_distributed() -> None:
    """Clean up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _benchmark_allreduce(
    rank: int,
    cfg: BenchConfig,
    device: torch.device,
) -> BenchResult:
    """
    Benchmark all-reduce operation.

    Returns the result from rank 0.
    """
    # Calculate the number of elements needed for the specified data size
    # data_size_mb is in megabytes, so convert to bytes then to float32 elements
    data_size_bytes = cfg.data_size_mb * 1024 * 1024
    num_elements = int(data_size_bytes / 4)  # float32 is 4 bytes

    # Create tensor to all-reduce
    tensor = torch.randn(num_elements, device=device, dtype=torch.float32)

    # Warmup iterations
    for _ in range(cfg.warmup_iterations):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Synchronize before benchmarking
    if device.type == "cuda":
        torch.cuda.synchronize()
    dist.barrier()

    # Benchmark iterations
    times_ms = []
    for _ in range(cfg.benchmark_iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()

        times_ms.append((end_time - start_time) * 1000)

    # Compute statistics
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    mean_ms = statistics.mean(times_ms)
    stdev_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

    return BenchResult(
        backend=cfg.backend,
        device_type=cfg.device_type,
        data_size_mb=cfg.data_size_mb,
        world_size=cfg.world_size,
        rank=rank,
        min_ms=min_ms,
        max_ms=max_ms,
        mean_ms=mean_ms,
        stdev_ms=stdev_ms,
    )


def _worker_process(
    rank: int,
    world_size: int,
    cfg: BenchConfig,
    results_queue,
) -> None:
    """Worker process for benchmarking."""
    try:
        device = _setup_distributed(rank, world_size, cfg.backend)
        result = _benchmark_allreduce(rank, cfg, device)
        results_queue.put(result)
    except Exception as e:
        logger.error(f"Rank {rank} failed with error: {e}")
        results_queue.put(None)
    finally:
        _cleanup_distributed()


def _run_benchmark(cfg: BenchConfig) -> BenchResult:
    """Run a single all-reduce benchmark with the given configuration."""
    world_size = cfg.world_size

    # Use a queue to collect results from all processes
    results_queue = mp.Queue()

    # Spawn worker processes
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=_worker_process, args=(rank, world_size, cfg, results_queue))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Collect results (rank 0 result should be representative)
    results = []
    while not results_queue.empty():
        result = results_queue.get()
        if result is not None:
            results.append(result)

    if not results:
        raise RuntimeError(f"No results collected for config: {cfg}")

    # Return the result from rank 0 (or the first available)
    return results[0]


def run_all_benchmarks() -> list[BenchResult]:
    """Run all benchmark configurations."""
    configs = []

    # Define benchmark configurations
    backends_devices = [
        ("gloo", "cpu"),
        ("nccl", "cuda"),
    ]
    data_sizes_mb = [1, 10, 100, 1024]  # 1MB, 10MB, 100MB, 1GB
    world_sizes = [2, 4, 6]

    for backend, device_type in backends_devices:
        # Skip NCCL on CPU or Gloo on GPU (not typical configurations)
        if backend == "nccl" and device_type == "cpu":
            continue
        if backend == "gloo" and device_type == "cuda":
            continue

        for data_size_mb in data_sizes_mb:
            for world_size in world_sizes:
                # Check GPU availability for NCCL
                if backend == "nccl":
                    if not torch.cuda.is_available():
                        logger.warning("CUDA not available, skipping NCCL benchmark")
                        continue
                    if torch.cuda.device_count() < world_size:
                        logger.warning(
                            f"Insufficient GPUs for world_size={world_size} "
                            f"(available: {torch.cuda.device_count()}), skipping"
                        )
                        continue

                cfg = BenchConfig(
                    backend=backend,
                    device_type=device_type,
                    data_size_mb=data_size_mb,
                    world_size=world_size,
                )
                configs.append(cfg)

    # Run benchmarks
    all_results = []
    total_configs = len(configs)

    for i, cfg in enumerate(configs):
        logger.info(
            f"[{i+1}/{total_configs}] Running benchmark: "
            f"backend={cfg.backend}, device={cfg.device_type}, "
            f"data_size={cfg.data_size_mb}MB, world_size={cfg.world_size}"
        )

        try:
            result = _run_benchmark(cfg)
            all_results.append(result)
            logger.info(
                f"  Result: mean={result.mean_ms:.3f}ms (±{result.stdev_ms:.3f}ms), "
                f"min={result.min_ms:.3f}ms, max={result.max_ms:.3f}ms"
            )
        except Exception as e:
            logger.error(f"  Failed: {e}")

    return all_results


def save_results_json(results: list[BenchResult], output_path: str) -> None:
    """Save benchmark results to JSON file."""
    data = [
        {
            "backend": r.backend,
            "device_type": r.device_type,
            "data_size_mb": r.data_size_mb,
            "world_size": r.world_size,
            "rank": r.rank,
            "min_ms": r.min_ms,
            "max_ms": r.max_ms,
            "mean_ms": r.mean_ms,
            "stdev_ms": r.stdev_ms,
        }
        for r in results
    ]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved results to {output_path}")


def print_results_table(results: list[BenchResult]) -> None:
    """Print results as a formatted table."""
    print("\n" + "=" * 100)
    print("All-Reduce Benchmark Results")
    print("=" * 100)
    print(
        f"{'Backend':<8} {'Device':<6} {'Data Size':<12} {'World Size':<12} "
        f"{'Mean (ms)':<12} {'Std Dev':<12} {'Min':<10} {'Max':<10}"
    )
    print("-" * 100)

    for result in sorted(
        results, key=lambda r: (r.backend, r.device_type, r.data_size_mb, r.world_size)
    ):
        print(
            f"{result.backend:<8} {result.device_type:<6} {result.data_size_mb:>10.1f}MB  "
            f"{result.world_size:>10}   {result.mean_ms:>10.3f}   "
            f"{result.stdev_ms:>10.3f}   {result.min_ms:>8.3f}   {result.max_ms:>8.3f}"
        )

    print("=" * 100 + "\n")


def generate_plots(results: list[BenchResult], output_dir: str = ".") -> None:
    """Generate plots from benchmark results."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Plot 1: Mean time vs data size for different world sizes (Gloo CPU)
    plt.figure(figsize=(12, 6))
    gloo_results = [r for r in results if r.backend == "gloo" and r.device_type == "cpu"]
    for world_size in sorted(set(r.world_size for r in gloo_results)):
        data = sorted(
            [r for r in gloo_results if r.world_size == world_size],
            key=lambda r: r.data_size_mb,
        )
        if data:
            sizes = [r.data_size_mb for r in data]
            times = [r.mean_ms for r in data]
            plt.plot(sizes, times, marker="o", label=f"World Size: {world_size}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Data Size (MB)")
    plt.ylabel("Mean All-Reduce Time (ms)")
    plt.title("Gloo Backend (CPU) - All-Reduce Time vs Data Size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "allreduce_gloo_cpu.png", dpi=150)
    logger.info(f"Saved plot to {output_dir / 'allreduce_gloo_cpu.png'}")
    plt.close()

    # Plot 2: Mean time vs data size for different world sizes (NCCL GPU)
    nccl_results = [r for r in results if r.backend == "nccl" and r.device_type == "cuda"]
    if nccl_results:
        plt.figure(figsize=(12, 6))
        for world_size in sorted(set(r.world_size for r in nccl_results)):
            data = sorted(
                [r for r in nccl_results if r.world_size == world_size],
                key=lambda r: r.data_size_mb,
            )
            if data:
                sizes = [r.data_size_mb for r in data]
                times = [r.mean_ms for r in data]
                plt.plot(sizes, times, marker="s", label=f"World Size: {world_size}")

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Data Size (MB)")
        plt.ylabel("Mean All-Reduce Time (ms)")
        plt.title("NCCL Backend (GPU) - All-Reduce Time vs Data Size")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "allreduce_nccl_gpu.png", dpi=150)
        logger.info(f"Saved plot to {output_dir / 'allreduce_nccl_gpu.png'}")
        plt.close()

    # Plot 3: Comparison of backends at 100MB
    plt.figure(figsize=(12, 6))
    comparison_size = 100  # 100MB
    backends_data = {}

    for r in results:
        if r.data_size_mb == comparison_size:
            key = f"{r.backend.upper()} ({r.device_type.upper()})"
            if key not in backends_data:
                backends_data[key] = {}
            backends_data[key][r.world_size] = r.mean_ms

    x = np.arange(len(sorted(set(r.world_size for r in results))))
    width = 0.35

    for i, (backend, data) in enumerate(sorted(backends_data.items())):
        world_sizes = sorted(data.keys())
        times = [data[ws] for ws in world_sizes]
        plt.bar(x + i * width, times, width, label=backend)

    plt.xlabel("World Size")
    plt.ylabel("Mean All-Reduce Time (ms)")
    plt.title(f"Backend Comparison (Data Size: {comparison_size}MB)")
    plt.xticks(x + width / 2, sorted(set(r.world_size for r in results)))
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "allreduce_backend_comparison.png", dpi=150)
    logger.info(f"Saved plot to {output_dir / 'allreduce_backend_comparison.png'}")
    plt.close()


def main():
    """Main entry point."""
    # Set up multiprocessing context
    mp.set_start_method("spawn", force=True)

    logger.info("Starting all-reduce benchmarks...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")

    # Run all benchmarks
    results = run_all_benchmarks()

    if not results:
        logger.error("No benchmark results collected")
        return

    # Print results table
    print_results_table(results)

    # Save results to JSON
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    save_results_json(results, str(output_dir / "allreduce_results.json"))

    # Generate plots
    generate_plots(results, output_dir)

    logger.info(f"Benchmark complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
