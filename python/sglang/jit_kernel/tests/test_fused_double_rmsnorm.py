import itertools
import sys

import pytest
import torch

from sglang.jit_kernel.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)

EPS = 1e-6
DEVICE = "cuda"
DTYPES = [torch.float16, torch.bfloat16]

BS_LIST = [2**n for n in range(0, 14)]
BS_LIST += [x + 1 + i for i, x in enumerate(BS_LIST)]
BS_LIST = get_ci_test_range(BS_LIST, [1, 9, 256, 4109])
HIDDEN_SIZE_LIST = get_ci_test_range(
    [512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192],
    [512, 2048, 8192],
)


def reference_double_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    eps: float,
):
    """Pure PyTorch reference in float32 for numerical accuracy."""
    x = input.float()
    rms1 = x.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    normed1 = x * rms1 * weight1.float()
    mid = normed1 + residual.float()
    rms2 = mid.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    output = mid * rms2 * weight2.float()
    return output.to(input.dtype), mid.to(input.dtype)


@pytest.mark.parametrize(
    "batch_size,hidden_size",
    list(itertools.product(BS_LIST, HIDDEN_SIZE_LIST)),
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_double_rmsnorm(
    batch_size: int, hidden_size: int, dtype: torch.dtype
) -> None:
    from sglang.jit_kernel.norm import fused_double_rmsnorm

    input = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    residual = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    weight1 = torch.randn(hidden_size, device=DEVICE, dtype=dtype)
    weight2 = torch.randn(hidden_size, device=DEVICE, dtype=dtype)

    output_ref, mid_ref = reference_double_rmsnorm(
        input, residual, weight1, weight2, EPS
    )
    output_jit, mid_jit = fused_double_rmsnorm(input, residual, weight1, weight2, EPS)

    torch.testing.assert_close(output_jit, output_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(mid_jit, mid_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
