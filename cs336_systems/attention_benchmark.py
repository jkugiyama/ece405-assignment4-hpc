"""
Benchmark scaled_dot_product_attention at various (d_k, seq_len) configurations.

Setup:
  - Batch size fixed at 8, no head dimension (Q/K/V shape: [batch, seq_len, d_k])
  - d_k  in [16, 32, 64, 128]
  - seq_len in [256, 1024, 4096, 8192, 16384]
  - 3 warmup steps + 100 benchmark steps
  - Memory (GPU) measured right after the forward pass, before backward
  - torch.cuda.synchronize() called after every forward/backward
"""

import itertools
import timeit
from logging import INFO, basicConfig, getLogger

import torch

from cs336_basics.model import scaled_dot_product_attention

logger = getLogger(__name__)
basicConfig(level=INFO)

BATCH_SIZE   = 8
D_K_VALUES   = [16, 32, 64, 128]
SEQ_LEN_VALUES = [256, 1024, 4096, 8192, 16384]
WARM_UP_STEPS   = 3
BENCHMARK_STEPS = 10 if not torch.cuda.is_available() else 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_CUDA = DEVICE == "cuda"


def sync():
    if USE_CUDA:
        torch.cuda.synchronize()


def cal_mem(d_k: int, seq_len: int, batch_size: int = BATCH_SIZE) -> float:
    """Theoretical memory (MB, fp32) for one attention forward pass.

    Activations saved for backward:
      Q, K, V       : 3 * batch * seq_len * d_k  floats
      attn_scores   : batch * seq_len * seq_len   floats  (pre-softmax logits)
      attn_weights  : batch * seq_len * seq_len   floats  (post-softmax)
      output        : batch * seq_len * d_k       floats
    """
    bytes_per_float = 4
    qkv    = 3 * batch_size * seq_len * d_k
    scores = batch_size * seq_len * seq_len        # pre-softmax
    weights = batch_size * seq_len * seq_len       # post-softmax (saved for bwd)
    output = batch_size * seq_len * d_k
    total_bytes = (qkv + scores + weights + output) * bytes_per_float
    return total_bytes / (1024 ** 2)


def benchmark(d_k: int, seq_len: int):
    """Run warmup + benchmark steps.  Returns (fwd_ms, mem_mb, bwd_ms)."""

    shape = (BATCH_SIZE, seq_len, d_k)

    forward_time_sum  = 0.0
    backward_time_sum = 0.0
    init_mem_sum      = 0.0
    peak_mem_sum      = 0.0

    logger.info("Benchmarking d_k=%d  seq_len=%d ...", d_k, seq_len)

    for i in range(WARM_UP_STEPS + BENCHMARK_STEPS):
        is_warmup = i < WARM_UP_STEPS
        step_label = f"warmup {i+1}" if is_warmup else f"step {i - WARM_UP_STEPS + 1}"
        logger.info("  %s", step_label)

        Q = torch.randn(shape, device=DEVICE, requires_grad=True)
        K = torch.randn(shape, device=DEVICE, requires_grad=True)
        V = torch.randn(shape, device=DEVICE, requires_grad=True)

        if USE_CUDA:
            torch.cuda.reset_peak_memory_stats(DEVICE)
        init_mem = torch.cuda.memory_allocated(DEVICE) if USE_CUDA else 0

        # ---- forward ----
        t1 = timeit.default_timer()
        with (torch.cuda.nvtx.range("forward") if USE_CUDA else _nullctx()):
            outputs = scaled_dot_product_attention(Q, K, V)
            sync()
        t2 = timeit.default_timer()

        # memory right after forward, before backward
        peak_mem = torch.cuda.max_memory_allocated(DEVICE) if USE_CUDA else 0

        # ---- backward ----
        grad_out = torch.ones_like(outputs)
        t3 = timeit.default_timer()
        with (torch.cuda.nvtx.range("backward") if USE_CUDA else _nullctx()):
            outputs.backward(grad_out)
            sync()
        t4 = timeit.default_timer()

        del Q, K, V, outputs, grad_out

        if is_warmup:
            continue

        forward_time_sum  += t2 - t1
        backward_time_sum += t4 - t3
        init_mem_sum      += init_mem
        peak_mem_sum      += peak_mem

    avg_fwd_ms  = 1e3 * forward_time_sum  / BENCHMARK_STEPS
    avg_bwd_ms  = 1e3 * backward_time_sum / BENCHMARK_STEPS
    avg_mem_mb  = (peak_mem_sum / BENCHMARK_STEPS) / (1024 ** 2) if USE_CUDA else float("nan")

    logger.info(
        "  avg fwd=%.3f ms  bwd=%.3f ms  mem_before_bwd=%.1f MB",
        avg_fwd_ms, avg_bwd_ms, avg_mem_mb,
    )
    return avg_fwd_ms, avg_mem_mb, avg_bwd_ms


class _nullctx:
    """Minimal no-op context manager (avoids importing contextlib)."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


def main():
    logger.info("Device: %s", DEVICE)
    logger.info("Warmup=%d  Benchmark=%d", WARM_UP_STEPS, BENCHMARK_STEPS)

    # Start CUDA memory history recording once (GPU only)
    if USE_CUDA:
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    header = (
        f"{'d_k':>6} {'seq_len':>8} "
        f"{'fwd (ms)':>12} {'mem_pre_bwd (MB)':>18} {'bwd (ms)':>12} {'theory_mem (MB)':>16}"
    )
    sep = "-" * len(header)
    print("\n" + header)
    print(sep)

    for d_k, seq_len in itertools.product(D_K_VALUES, SEQ_LEN_VALUES):
        theory_mb = cal_mem(d_k, seq_len)
        try:
            fwd_ms, mem_mb, bwd_ms = benchmark(d_k, seq_len)
            print(
                f"{d_k:>6} {seq_len:>8} "
                f"{fwd_ms:>12.3f} {mem_mb:>18.1f} {bwd_ms:>12.3f} {theory_mb:>16.1f}"
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                print(
                    f"{d_k:>6} {seq_len:>8} "
                    f"{'OOM':>12} {'OOM':>18} {'OOM':>12} {theory_mb:>16.1f}"
                )
                if USE_CUDA:
                    torch.cuda.empty_cache()
            else:
                print(
                    f"{d_k:>6} {seq_len:>8} "
                    f"{'ERR':>12} {'ERR':>18} {'ERR':>12} {theory_mb:>16.1f}  ({e})"
                )
                if USE_CUDA:
                    torch.cuda.empty_cache()

    print(sep)

    if USE_CUDA:
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        logger.info("Memory snapshot saved to memory_snapshot.pickle")
    else:
        logger.info("(CPU run: no CUDA memory snapshot produced)")

    print("Done.")


if __name__ == "__main__":
    main()
