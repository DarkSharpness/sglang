"""Benchmark for DSV3 sparse indexer: SGL JIT kernel vs deep_gemm baseline.

Measures per-layer kernel latency of the paged MQA logits computation:
  logits[b, t] = sum_h( relu(q[b,h] . K_dequant[b,t,h]) * w[b,h] )

Uses CUDA graph capture (run_benchmark) for accurate kernel timing.
Runs NUM_LAYERS iterations per measurement to amortize L2 cache effects.

Scenarios:
  1. Decode: fixed-len, vary (seq_len, batch) grid
  2. Decode varlen: exponential-distributed seq_lens, vary (mean_len, batch) grid

Usage:
  CUDA_VISIBLE_DEVICES=7 python -m sglang.jit_kernel.benchmark.bench_dsv3_indexer
"""

from __future__ import annotations

import gc
import itertools

import deep_gemm
import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.jit_kernel.dsv3_indexer import (
    dsv3_indexer,
    get_indexer_metadata,
)
from sglang.jit_kernel.tests.test_dsv3_indexer import make_kv_cache

PAGE_SIZE = 64
NUM_HEADS = 64
HEAD_DIM = 128
SM_COUNT = deep_gemm.get_num_sms()

# Run multiple layers per measurement to flush L2 cache effects
NUM_LAYERS = 3

torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _make_data(batch_size: int, seq_lens_list: list[int]):
    """Create benchmark data for given per-batch sequence lengths."""
    gc.collect()
    torch.cuda.synchronize()
    device = "cuda"
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)

    pages_per_seq = [(sl + PAGE_SIZE - 1) // PAGE_SIZE for sl in seq_lens_list]
    total_pages = sum(pages_per_seq)
    max_pages = max(pages_per_seq)
    max_pages = (max_pages + 1) // 2 * 2

    q = torch.randn(
        batch_size, NUM_HEADS, HEAD_DIM, dtype=torch.float32, device=device
    ).to(torch.float8_e4m3fn)
    kv_cache = make_kv_cache(total_pages)
    weights = torch.randn(batch_size, NUM_HEADS, dtype=torch.float32, device=device)

    block_table = torch.zeros(batch_size, max_pages, dtype=torch.int32, device=device)
    offset = 0
    for b in range(batch_size):
        n = pages_per_seq[b]
        block_table[b, :n] = torch.arange(
            offset, offset + n, dtype=torch.int32, device=device
        )
        offset += n

    sm_map = get_indexer_metadata(seq_lens)
    max_seq_len = max_pages * PAGE_SIZE
    dg_metadata = deep_gemm.get_paged_mqa_logits_metadata(seq_lens, PAGE_SIZE, SM_COUNT)

    logits_buf = torch.empty(
        batch_size, max_seq_len, device=device, dtype=torch.float32
    )
    q_4d = q.unsqueeze(1)
    torch.cuda.synchronize()

    return {
        "q": q,
        "q_4d": q_4d,
        "kv_cache": kv_cache,
        "weights": weights,
        "seq_lens": seq_lens,
        "block_table": block_table,
        "sm_map": sm_map,
        "dg_metadata": dg_metadata,
        "max_seq_len": max_seq_len,
        "logits_buf": logits_buf,
    }


def _make_fixed(batch_size: int, seq_len: int):
    return _make_data(batch_size, [seq_len] * batch_size)


def _make_varlen_decode(batch_size: int, mean_seq_len: int):
    """Variable-length decode in [mean // 4, mean * 4], with exponential distribution."""
    torch.random.manual_seed(42)
    lo = max(PAGE_SIZE, mean_seq_len // 4)
    hi = mean_seq_len * 4
    seq_lens = torch.clamp(
        torch.round(torch.empty(batch_size).exponential_(1.0) * mean_seq_len),
        lo,
        hi,
    ).to(torch.int32)
    return _make_data(batch_size, seq_lens.tolist())


def _bench(d, provider):
    """Create a multi-layer benchmark function and run with cuda graph."""

    def fn():
        if provider == "sgl_jit":
            for _ in range(NUM_LAYERS):
                dsv3_indexer(
                    d["q"],
                    d["kv_cache"],
                    d["weights"],
                    d["seq_lens"],
                    d["block_table"],
                    sm_map=d["sm_map"],
                    max_model_len=d["max_seq_len"],
                    logits_out=d["logits_buf"],
                )
        else:
            for _ in range(NUM_LAYERS):
                deep_gemm.fp8_paged_mqa_logits(
                    d["q_4d"],
                    d["kv_cache"],
                    d["weights"],
                    d["seq_lens"],
                    d["block_table"],
                    d["dg_metadata"],
                    d["max_seq_len"],
                    clean_logits=False,
                )

    return run_benchmark(fn, scale=NUM_LAYERS)


# ---------------------------------------------------------------------------
# Benchmark configs
# ---------------------------------------------------------------------------

PROVIDERS = ["sgl_jit", "deep_gemm"]
NAMES = ["SGL JIT Indexer", "DeepGemm"]
STYLES = [("green", "-"), ("blue", "-")]

DECODE_BATCH_SIZES = get_benchmark_range(
    full_range=[1, 4, 8, 16, 32, 64, 128],
    ci_range=[4, 32],
)

CONTEXT_LENGTHS = get_benchmark_range(
    full_range=[4096, 8192, 16384, 32768, 65536, 131072],
    ci_range=[4096],
)

VARLEN_BATCH_SIZES = get_benchmark_range(
    full_range=[1, 4, 8, 16, 32, 64, 128, 256, 512],
    ci_range=[8, 64],
)

VARLEN_MEAN_SEQ_LENS = get_benchmark_range(
    full_range=[4096, 8192, 16384, 32768],
    ci_range=[4096],
)


# ---------------------------------------------------------------------------
# 1. Decode: fixed-len, vary (seq_len, batch) grid
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len", "batch_size"],
        x_vals=list(itertools.product(CONTEXT_LENGTHS, DECODE_BATCH_SIZES)),
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="dsv3-indexer-decode",
        args={},
    )
)
def bench_decode(batch_size: int, seq_len: int, provider: str):
    d = _make_fixed(batch_size, seq_len)
    return _bench(d, provider)


# ---------------------------------------------------------------------------
# 2. Decode varlen: exponential-distributed seq_lens, vary (mean_len, batch)
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["mean_seq_len", "batch_size"],
        x_vals=list(itertools.product(VARLEN_MEAN_SEQ_LENS, VARLEN_BATCH_SIZES)),
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="dsv3-indexer-decode-varlen",
        args={},
    )
)
def bench_decode_varlen(provider, batch_size, mean_seq_len):
    d = _make_varlen_decode(batch_size, mean_seq_len)
    return _bench(d, provider)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print(f"DSV3 Indexer Benchmark (layers={NUM_LAYERS}, cuda graph)")
    print("=" * 70)

    print("\n--- 1. Decode (fixed-len, vary seq_len x batch) ---")
    bench_decode.run(print_data=True)

    print("\n--- 2. Decode varlen (exponential, vary mean x batch) ---")
    bench_decode_varlen.run(print_data=True)
