import itertools

import torch
import triton
import triton.testing
from flashinfer.norm import rmsnorm

from sglang.jit_kernel.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.jit_kernel.norm import fused_add_rmsnorm as jit_fused_add_rmsnorm
from sglang.jit_kernel.norm import fused_double_rmsnorm as jit_fused_double_rmsnorm
from sglang.srt.layers.elementwise import (
    fused_dual_residual_rmsnorm as triton_fused_double_rmsnorm,
)

DTYPE = torch.bfloat16
DEVICE = "cuda"

BS_LIST = get_benchmark_range(
    full_range=[2**n for n in range(0, 14)],
    ci_range=[16, 32],
)
HIDDEN_SIZE_LIST = [2048, 4096, 8192]

LINE_VALS = ["triton", "jit_baseline", "jit_fused"]
LINE_NAMES = [
    "Triton fused",
    "JIT fused_add_rmsnorm + rmsnorm",
    "JIT fused_double_rmsnorm",
]
STYLES = [("blue", "--"), ("red", "-."), ("green", "-")]
NUM_LAYERS = 4  # avoid L2 effect

configs = list(itertools.product(HIDDEN_SIZE_LIST, BS_LIST))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["hidden_size", "batch_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="fused-double-rmsnorm-performance",
        args={},
    )
)
def benchmark_fused_double_rmsnorm(hidden_size: int, batch_size: int, provider: str):
    x = torch.randn((NUM_LAYERS, batch_size, hidden_size), dtype=DTYPE, device=DEVICE)
    residual = torch.randn_like(x)
    weight1 = torch.randn((NUM_LAYERS, hidden_size), dtype=DTYPE, device=DEVICE)
    weight2 = torch.randn_like(weight1)

    def f():
        if provider == "triton":
            for i in range(NUM_LAYERS):
                triton_fused_double_rmsnorm(
                    x[i], residual[i], weight1[i], weight2[i], eps=1e-6
                )
        elif provider == "jit_baseline":
            for i in range(NUM_LAYERS):
                # fused_add_rmsnorm writes normed1 into x[i] and x[i]+residual into residual[i]
                jit_fused_add_rmsnorm(x[i], residual[i], weight1[i], eps=1e-6)
                rmsnorm(residual[i], weight2[i], eps=1e-6)
        else:
            for i in range(NUM_LAYERS):
                jit_fused_double_rmsnorm(
                    x[i], residual[i], weight1[i], weight2[i], eps=1e-6
                )

    return run_benchmark(f, scale=NUM_LAYERS)


if __name__ == "__main__":
    print("Benchmarking fused double rmsnorm...")
    benchmark_fused_double_rmsnorm.run(print_data=True)
