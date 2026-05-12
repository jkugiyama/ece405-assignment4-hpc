import argparse
import math
from dataclasses import dataclass
from typing import Callable

import torch

try:
    import triton
except ImportError:  # pragma: no cover
    triton = None


def pytorch_attention_no_flash(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> torch.Tensor:
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    scores = (q @ k.transpose(-2, -1)) * scale

    if is_causal:
        t_q = q.shape[-2]
        t_k = k.shape[-2]
        q_idx = torch.arange(t_q, device=q.device)[:, None]
        k_idx = torch.arange(t_k, device=q.device)[None, :]
        scores = scores.masked_fill(q_idx < k_idx, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    return probs @ v


@dataclass
class BenchResult:
    impl: str
    dtype: str
    d_model: int
    seq_len: int
    forward_ms: float
    backward_ms: float
    e2e_ms: float


def bench_impl(
    impl_name: str,
    impl: Callable,
    *,
    seq_len: int,
    d_model: int,
    dtype: torch.dtype,
    warmup: int,
    rep: int,
) -> BenchResult:
    device = "cuda"
    batch_size = 1

    q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

    # Fixed upstream gradient tensor, generated before timing.
    out_seed = impl(q, k, v, True)
    do = torch.randn_like(out_seed)

    def forward_only():
        return impl(q, k, v, True)

    # Backward-only: build one graph once, then repeatedly traverse it.
    out_for_backward = impl(q, k, v, True)

    def backward_only():
        torch.autograd.grad(
            out_for_backward,
            (q, k, v),
            grad_outputs=do,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

    def end_to_end():
        out = impl(q, k, v, True)
        torch.autograd.grad(
            out,
            (q, k, v),
            grad_outputs=do,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )

    forward_ms = triton.testing.do_bench(forward_only, warmup=warmup, rep=rep)
    backward_ms = triton.testing.do_bench(backward_only, warmup=warmup, rep=rep)
    e2e_ms = triton.testing.do_bench(end_to_end, warmup=warmup, rep=rep)

    return BenchResult(
        impl=impl_name,
        dtype=str(dtype).replace("torch.", ""),
        d_model=d_model,
        seq_len=seq_len,
        forward_ms=float(forward_ms),
        backward_ms=float(backward_ms),
        e2e_ms=float(e2e_ms),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Triton FlashAttention-2 vs regular PyTorch attention")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations per benchmark")
    parser.add_argument("--rep", type=int, default=100, help="Measured iterations per benchmark")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=65536,
        choices=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
        help="Upper bound for sequence length sweep",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Optional path to write CSV results",
    )
    return parser.parse_args()


def print_table(results: list[BenchResult]) -> None:
    header = (
        f"{'impl':>10} {'dtype':>8} {'d_model':>8} {'seq_len':>8} "
        f"{'forward_ms':>12} {'backward_ms':>12} {'e2e_ms':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.impl:>10} {r.dtype:>8} {r.d_model:>8} {r.seq_len:>8} "
            f"{r.forward_ms:>12.3f} {r.backward_ms:>12.3f} {r.e2e_ms:>12.3f}"
        )


def maybe_write_csv(results: list[BenchResult], output_csv: str) -> None:
    if not output_csv:
        return
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("impl,dtype,d_model,seq_len,forward_ms,backward_ms,e2e_ms\n")
        for r in results:
            f.write(
                f"{r.impl},{r.dtype},{r.d_model},{r.seq_len},"
                f"{r.forward_ms:.6f},{r.backward_ms:.6f},{r.e2e_ms:.6f}\n"
            )


def main() -> None:
    args = parse_args()

    if triton is None:
        raise RuntimeError("Triton is required for this benchmark. Install triton and run on a single H100.")

    from cs336_systems.flash_attention_triton import FlashAttnTritonFunc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark. Run on a single H100.")

    device_name = torch.cuda.get_device_name(0)
    print(f"Running on GPU: {device_name}")

    seq_lens = [2**p for p in range(7, 17) if 2**p <= args.max_seq_len]
    d_models = [16, 32, 64, 128]
    dtypes = [torch.float32]

    results: list[BenchResult] = []

    for dtype in dtypes:
        for d_model in d_models:
            for seq_len in seq_lens:
                try:
                    triton_res = bench_impl(
                        "triton",
                        FlashAttnTritonFunc.apply,
                        seq_len=seq_len,
                        d_model=d_model,
                        dtype=dtype,
                        warmup=args.warmup,
                        rep=args.rep,
                    )
                    torch_res = bench_impl(
                        "pytorch",
                        pytorch_attention_no_flash,
                        seq_len=seq_len,
                        d_model=d_model,
                        dtype=dtype,
                        warmup=args.warmup,
                        rep=args.rep,
                    )
                    results.extend([triton_res, torch_res])
                except torch.cuda.OutOfMemoryError:
                    print(
                        "OOM for config:",
                        f"dtype={str(dtype).replace('torch.', '')}",
                        f"d_model={d_model}",
                        f"seq_len={seq_len}",
                    )
                    torch.cuda.empty_cache()

    print_table(results)
    maybe_write_csv(results, args.output_csv)


if __name__ == "__main__":
    main()
