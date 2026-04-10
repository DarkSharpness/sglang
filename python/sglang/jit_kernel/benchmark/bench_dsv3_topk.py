"""Benchmark for DSV3 cluster-based radix top-K vs torch.topk vs sgl_kernel AOT.

Uses CUDA graph capture (run_benchmark) for accurate kernel timing.

Scenarios:
  1. Decode fixed-len: vary (seq_len, batch) grid, K=2048
  2. K sweep: vary (K, batch) grid, seq_len=16384
  3. Decode varlen: exponential-distributed seq_lens, vary (mean_len, batch) grid
  4. Cluster sweep: vary num_clusters at representative (seq_len, batch) grid

Usage:
  CUDA_VISIBLE_DEVICES=7 python -m sglang.jit_kernel.benchmark.bench_dsv3_topk
"""

from __future__ import annotations

import gc
import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.jit_kernel.dsv3_indexer import dsv3_topk, get_topk_num_clusters

# sgl_kernel AOT topk (K=2048 only)
try:
    import sgl_kernel  # noqa: F401

    _sgl_fast_topk = torch.ops.sgl_kernel.fast_topk  # pre-allocated interface
    HAS_SGL_KERNEL = True
except Exception:
    HAS_SGL_KERNEL = False

torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _make_data(batch_size: int, seq_lens_list: list[int], K: int, num_clusters: int):
    """Pre-allocate all tensors for CUDA graph capture."""
    gc.collect()
    torch.cuda.synchronize()
    device = "cuda"

    max_len = max(seq_lens_list)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
    logits = torch.randn(batch_size, max_len, device=device, dtype=torch.float32)

    # JIT radix topk buffers
    indices_out = torch.empty(batch_size, K, dtype=torch.int32, device=device)
    chunk = (max_len + num_clusters - 1) // num_clusters
    ov_stride = max(2048, chunk // 4)
    overflow_buf = torch.empty(
        batch_size, num_clusters * ov_stride * 4, dtype=torch.int32, device=device
    )

    # AOT topk buffer (K=2048 only)
    aot_indices = torch.empty(batch_size, 2048, dtype=torch.int32, device=device)

    torch.cuda.synchronize()

    return {
        "logits": logits,
        "seq_lens": seq_lens,
        "max_len": max_len,
        "K": K,
        "nc": num_clusters,
        "indices_out": indices_out,
        "overflow_buf": overflow_buf,
        "aot_indices": aot_indices,
    }


def _make_fixed(batch_size: int, seq_len: int, K: int, num_clusters: int | None = None):
    nc = num_clusters or get_topk_num_clusters(batch_size, seq_len)
    return _make_data(batch_size, [seq_len] * batch_size, K, nc)


def _make_varlen(batch_size: int, mean_seq_len: int, K: int):
    torch.random.manual_seed(42)
    lo = max(K + 1, mean_seq_len // 4)
    hi = mean_seq_len * 4
    sl = torch.clamp(
        torch.round(torch.empty(batch_size).exponential_(1.0) * mean_seq_len), lo, hi
    ).to(torch.int32)
    max_len = int(sl.max().item())
    nc = get_topk_num_clusters(batch_size, max_len)
    return _make_data(batch_size, sl.tolist(), K, nc)


def _bench(d, provider):
    def fn():
        if provider == "radix_topk":
            dsv3_topk(
                d["logits"],
                d["seq_lens"],
                d["K"],
                num_clusters=d["nc"],
                indices_out=d["indices_out"],
                overflow_buf=d["overflow_buf"],
            )
        elif provider == "sgl_aot":
            _sgl_fast_topk(d["logits"], d["aot_indices"], d["seq_lens"], None)
        else:
            torch.topk(d["logits"][:, : d["max_len"]], d["K"], dim=1)

    return run_benchmark(fn)


# ---------------------------------------------------------------------------
# Benchmark configs
# ---------------------------------------------------------------------------

# 3-provider list for K=2048 scenarios (includes AOT if available)
PROVIDERS_3 = ["radix_topk", "torch_topk"] + (["sgl_aot"] if HAS_SGL_KERNEL else [])
NAMES_3 = ["Cluster Radix TopK", "torch.topk"] + (
    ["sgl_kernel AOT"] if HAS_SGL_KERNEL else []
)
STYLES_3 = [("green", "-"), ("blue", "-")] + ([("red", "-")] if HAS_SGL_KERNEL else [])

# 2-provider list for K-sweep (AOT only supports K=2048)
PROVIDERS_2 = ["radix_topk", "torch_topk"]
NAMES_2 = ["Cluster Radix TopK", "torch.topk"]
STYLES_2 = [("green", "-"), ("blue", "-")]

BATCH_SIZES = get_benchmark_range(
    full_range=[1, 4, 8, 16, 32, 64, 128], ci_range=[4, 32]
)
CONTEXT_LENGTHS = get_benchmark_range(
    full_range=[4096, 8192, 16384, 32768, 65536, 131072], ci_range=[4096]
)
K_VALUES = get_benchmark_range(full_range=[512, 1024, 2048], ci_range=[2048])
VARLEN_BATCH_SIZES = get_benchmark_range(
    full_range=[1, 4, 8, 16, 32, 64, 128, 256], ci_range=[8, 64]
)
VARLEN_MEAN_SEQ_LENS = get_benchmark_range(
    full_range=[4096, 8192, 16384, 32768, 65536], ci_range=[4096]
)


# ---------------------------------------------------------------------------
# 1. Decode fixed-len, K=2048
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len", "batch_size"],
        x_vals=list(itertools.product(CONTEXT_LENGTHS, BATCH_SIZES)),
        line_arg="provider",
        line_vals=PROVIDERS_3,
        line_names=NAMES_3,
        styles=STYLES_3,
        ylabel="us",
        plot_name="dsv3-topk-decode",
        args={},
    )
)
def bench_decode(batch_size: int, seq_len: int, provider: str):
    d = _make_fixed(batch_size, seq_len, K=2048)
    return _bench(d, provider)


# ---------------------------------------------------------------------------
# 2. K sweep, seq_len=16384  (radix + torch only; AOT is K=2048-only)
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["K", "batch_size"],
        x_vals=list(itertools.product(K_VALUES, BATCH_SIZES)),
        line_arg="provider",
        line_vals=PROVIDERS_2,
        line_names=NAMES_2,
        styles=STYLES_2,
        ylabel="us",
        plot_name="dsv3-topk-k-sweep",
        args={},
    )
)
def bench_k_sweep(batch_size: int, K: int, provider: str):
    d = _make_fixed(batch_size, 16384, K=K)
    return _bench(d, provider)


# ---------------------------------------------------------------------------
# 3. Decode varlen, K=2048
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["mean_seq_len", "batch_size"],
        x_vals=list(itertools.product(VARLEN_MEAN_SEQ_LENS, VARLEN_BATCH_SIZES)),
        line_arg="provider",
        line_vals=PROVIDERS_3,
        line_names=NAMES_3,
        styles=STYLES_3,
        ylabel="us",
        plot_name="dsv3-topk-decode-varlen",
        args={},
    )
)
def bench_varlen(provider: str, batch_size: int, mean_seq_len: int):
    d = _make_varlen(batch_size, mean_seq_len, K=2048)
    return _bench(d, provider)


# ---------------------------------------------------------------------------
# 4. Cluster sweep: each nc is a provider line, vary (seq_len, batch)
# ---------------------------------------------------------------------------

CLUSTER_SIZES = [1, 2, 4, 8]
CLUSTER_PROVIDERS = [f"nc={nc}" for nc in CLUSTER_SIZES]
CLUSTER_STYLES = [("red", "-"), ("orange", "-"), ("green", "-"), ("blue", "-")]

CLUSTER_BATCH_SIZES = get_benchmark_range(
    full_range=[2**n for n in range(10)], ci_range=[1, 32, 256]
)
CLUSTER_SEQ_LENS = get_benchmark_range(
    full_range=[8192, 16384, 32768, 65536, 131072], ci_range=[8192]
)


def _bench_cluster(d, provider):
    """Benchmark a single cluster-size provider."""

    def fn():
        dsv3_topk(
            d["logits"],
            d["seq_lens"],
            d["K"],
            num_clusters=d["nc"],
            indices_out=d["indices_out"],
            overflow_buf=d["overflow_buf"],
        )

    return run_benchmark(fn)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len", "batch_size"],
        x_vals=list(itertools.product(CLUSTER_SEQ_LENS, CLUSTER_BATCH_SIZES)),
        line_arg="provider",
        line_vals=CLUSTER_PROVIDERS,
        line_names=[f"nc={nc}" for nc in CLUSTER_SIZES],
        styles=CLUSTER_STYLES,
        ylabel="us",
        plot_name="dsv3-topk-cluster-sweep",
        args={},
    )
)
def bench_cluster_sweep(batch_size: int, seq_len: int, provider: str):
    nc = int(provider.split("=")[1])
    d = _make_fixed(batch_size, seq_len, K=2048, num_clusters=nc)
    return _bench_cluster(d, provider)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("DSV3 TopK Benchmark (cuda graph)")
    print(f"  sgl_kernel AOT: {'available' if HAS_SGL_KERNEL else 'not found'}")
    print("=" * 70)

    print("\n--- 1. Decode (fixed-len, vary seq_len x batch, K=2048) ---")
    bench_decode.run(print_data=True)

    print("\n--- 2. K sweep (vary K x batch, seq_len=16384) ---")
    bench_k_sweep.run(print_data=True)

    print("\n--- 3. Decode varlen (exponential, vary mean x batch, K=2048) ---")
    bench_varlen.run(print_data=True)

    print("\n--- 4. Cluster sweep (vary nc, K=2048) ---")
    bench_cluster_sweep.run(print_data=True)
