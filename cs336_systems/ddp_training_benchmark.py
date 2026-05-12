"""
Benchmarking script for naive DDP training.

Measures:
- Total time per training step
- Time spent on gradient communication (all-reduce operations)
- Breakdown of compute vs communication overhead

Runs with 2 GPUs on a single node using the XL model size.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from cs336_systems.ddp_individual_parameters import DDPIndividualParameters

try:
    from cs336_basics.model import BasicsTransformerLM
    from cs336_basics.nn_utils import cross_entropy
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parent.parent
    basics_src = repo_root / "cs336-basics"
    if basics_src.exists():
        sys.path.insert(0, str(basics_src))
    from cs336_basics.model import BasicsTransformerLM
    from cs336_basics.nn_utils import cross_entropy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Model configuration from benchmark.py
MODEL_CONFIGS = {
    "xl": {
        "d_model": 1600,
        "d_ff": 6400,
        "num_layers": 48,
        "num_heads": 25,
    }
}

DEFAULT_VOCAB_SIZE = 50257
DEFAULT_BATCH_SIZE = 8
DEFAULT_CONTEXT_LENGTH = 256


def setup_process_group(rank: int, world_size: int, backend: str = "nccl"):
    """Initialize the distributed process group."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12390")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("No CUDA devices available")
        local_rank = rank % device_count
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
        backend = "gloo"

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def cleanup_process_group():
    """Cleanup the distributed process group."""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def create_model(
    config: dict,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    device: str = "cuda",
) -> nn.Module:
    """Create a BasicsTransformerLM model."""
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        rope_theta=10000.0,
    )
    return model.to(device)


def make_batch(
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a random batch of training data."""
    inputs = torch.randint(
        0, vocab_size, (batch_size, context_length), device=device
    )
    targets = torch.randint(
        0, vocab_size, (batch_size, context_length), device=device
    )
    return inputs, targets


def benchmark_ddp_training(
    rank: int,
    world_size: int,
    model_size: str = "xl",
    num_steps: int = 20,
    warmup_steps: int = 5,
    batch_size: int = DEFAULT_BATCH_SIZE,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    results_file: str = "ddp_benchmark_results.json",
):
    """
    Benchmark naive DDP training on a single node with 2 GPUs.

    Measures:
    - Total time per training step
    - Time spent on gradient communication
    - Communication overhead as percentage of total time
    """
    # Setup
    device = setup_process_group(rank, world_size)
    torch.manual_seed(42 + rank)  # Ensure different initialization per rank

    try:
        # Create model
        config = MODEL_CONFIGS[model_size]
        model = create_model(config, context_length=context_length, device=device)
        
        # Wrap with naive DDP
        ddp_model = DDPIndividualParameters(model)
        
        # Optimizer
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
        
        # Metrics collection
        step_times = []
        compute_times = []
        comm_times = []
        
        # Synchronize before starting
        dist.barrier()
        
        if rank == 0:
            logger.info(f"Benchmarking {model_size.upper()} model on {world_size} GPUs")
            logger.info(f"Model config: {config}")
            logger.info(f"Batch size: {batch_size}, Context length: {context_length}")
            logger.info(f"Warmup steps: {warmup_steps}, Benchmark steps: {num_steps}")
        
        # Training loop
        total_steps = warmup_steps + num_steps
        for step in range(total_steps):
            # Create batch
            inputs, targets = make_batch(
                batch_size, context_length, DEFAULT_VOCAB_SIZE, device
            )
            
            # Forward pass - time this as "compute"
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            compute_start = time.perf_counter()
            
            logits = ddp_model(inputs)
            loss = cross_entropy(logits, targets)
            
            # Backward pass - communication happens here asynchronously
            loss.backward()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            compute_end = time.perf_counter()
            compute_time = compute_end - compute_start
            
            # Finish gradient synchronization (wait for all-reduce to complete)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            comm_start = time.perf_counter()
            
            ddp_model.finish_gradient_synchronization()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            comm_end = time.perf_counter()
            comm_time = comm_end - comm_start
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            total_time = compute_time + comm_time
            
            # Only record times after warmup
            if step >= warmup_steps:
                step_times.append(total_time)
                compute_times.append(compute_time)
                comm_times.append(comm_time)
            
            if rank == 0 and (step % 5 == 0 or step < warmup_steps):
                logger.info(
                    f"Step {step:3d} | Total: {total_time*1000:7.2f}ms | "
                    f"Compute: {compute_time*1000:7.2f}ms | Comm: {comm_time*1000:7.2f}ms | "
                    f"Comm %: {(comm_time/total_time)*100:5.1f}%"
                )
        
        # Cleanup
        dist.barrier()
        cleanup_process_group()
        
        # Rank 0 collects and reports results
        if rank == 0:
            # Compute statistics
            results = {
                "model_size": model_size,
                "num_gpus": world_size,
                "batch_size": batch_size,
                "context_length": context_length,
                "model_config": config,
                "num_benchmark_steps": num_steps,
                "warmup_steps": warmup_steps,
                "metrics": {
                    "total_time_per_step_ms": {
                        "mean": statistics.mean(step_times) * 1000,
                        "stdev": statistics.stdev(step_times) * 1000 if len(step_times) > 1 else 0.0,
                        "min": min(step_times) * 1000,
                        "max": max(step_times) * 1000,
                    },
                    "compute_time_per_step_ms": {
                        "mean": statistics.mean(compute_times) * 1000,
                        "stdev": statistics.stdev(compute_times) * 1000 if len(compute_times) > 1 else 0.0,
                        "min": min(compute_times) * 1000,
                        "max": max(compute_times) * 1000,
                    },
                    "communication_time_per_step_ms": {
                        "mean": statistics.mean(comm_times) * 1000,
                        "stdev": statistics.stdev(comm_times) * 1000 if len(comm_times) > 1 else 0.0,
                        "min": min(comm_times) * 1000,
                        "max": max(comm_times) * 1000,
                    },
                    "communication_fraction": {
                        "mean": statistics.mean([c/t for c, t in zip(comm_times, step_times)]),
                        "min": min([c/t for c, t in zip(comm_times, step_times)]),
                        "max": max([c/t for c, t in zip(comm_times, step_times)]),
                    },
                },
            }
            
            # Log results
            logger.info("\n" + "="*70)
            logger.info("BENCHMARKING RESULTS")
            logger.info("="*70)
            logger.info(f"Model: {model_size.upper()}")
            logger.info(f"World Size: {world_size} GPUs")
            logger.info(f"Batch Size: {batch_size}")
            logger.info(f"Context Length: {context_length}")
            logger.info("-"*70)
            
            metrics = results["metrics"]
            
            logger.info("Total Time per Step:")
            logger.info(f"  Mean:   {metrics['total_time_per_step_ms']['mean']:.2f} ms")
            logger.info(f"  Stdev:  {metrics['total_time_per_step_ms']['stdev']:.2f} ms")
            logger.info(f"  Min:    {metrics['total_time_per_step_ms']['min']:.2f} ms")
            logger.info(f"  Max:    {metrics['total_time_per_step_ms']['max']:.2f} ms")
            
            logger.info("\nCompute Time per Step:")
            logger.info(f"  Mean:   {metrics['compute_time_per_step_ms']['mean']:.2f} ms")
            logger.info(f"  Stdev:  {metrics['compute_time_per_step_ms']['stdev']:.2f} ms")
            logger.info(f"  Min:    {metrics['compute_time_per_step_ms']['min']:.2f} ms")
            logger.info(f"  Max:    {metrics['compute_time_per_step_ms']['max']:.2f} ms")
            
            logger.info("\nCommunication Time per Step:")
            logger.info(f"  Mean:   {metrics['communication_time_per_step_ms']['mean']:.2f} ms")
            logger.info(f"  Stdev:  {metrics['communication_time_per_step_ms']['stdev']:.2f} ms")
            logger.info(f"  Min:    {metrics['communication_time_per_step_ms']['min']:.2f} ms")
            logger.info(f"  Max:    {metrics['communication_time_per_step_ms']['max']:.2f} ms")
            
            logger.info("\nCommunication as Fraction of Total Time:")
            logger.info(f"  Mean:   {metrics['communication_fraction']['mean']*100:.1f}%")
            logger.info(f"  Min:    {metrics['communication_fraction']['min']*100:.1f}%")
            logger.info(f"  Max:    {metrics['communication_fraction']['max']*100:.1f}%")
            logger.info("="*70)
            
            # Save results to JSON
            output_path = Path(results_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nResults saved to {output_path}")
            
    except Exception as e:
        logger.error(f"Error in rank {rank}: {e}", exc_info=True)
        raise
    finally:
        if dist.is_initialized():
            cleanup_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark naive DDP training on a single node with 2 GPUs"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        default="xl",
        help="Model size to benchmark",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="Number of benchmark steps (after warmup)",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=None,
        help="Alias for --num-steps to match other benchmark scripts",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Alias for --warmup-steps to match other benchmark scripts",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help="Context length for the model",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Alias for --context-length to match other benchmark scripts",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results/ddp_benchmark_results.json",
        help="Output file for benchmark results",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="Number of GPUs to use",
    )
    
    args = parser.parse_args()

    warmup_steps = args.warmup if args.warmup is not None else args.warmup_steps
    num_steps = args.rep if args.rep is not None else args.num_steps
    context_length = args.max_seq_len if args.max_seq_len is not None else args.context_length
    
    # Use torch.multiprocessing to spawn processes for each rank
    mp.spawn(
        benchmark_ddp_training,
        args=(
            args.world_size,
            args.model_size,
            num_steps,
            warmup_steps,
            args.batch_size,
            context_length,
            args.output,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
