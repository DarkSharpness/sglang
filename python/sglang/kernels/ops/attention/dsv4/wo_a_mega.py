"""Fused inverse-RoPE + grouped WO-A BF16 GEMM + MXFP8 quantization (SM100).

Collapses the three launches of the DSV4 decode/verify path -- ``fused_rope_inplace``,
``_wo_a_partial`` and ``_wo_a_reduce_quant`` -- into one cluster-launched kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from sglang.kernels.jit.utils import cache_once, cuda_stubs_dir, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

GROUPS = 2
GROUP_K = 4096
RANK = 1024
N_OUT = GROUPS * RANK
SCALE_BYTES = 8192


@cache_once
def _jit_module(max_tokens: int) -> Module:
    # Arch gating has to happen before load_jit: tcgen05 will not even compile
    # below SM100, and the ptxas error it produces is unreadable.
    major, minor = torch.cuda.get_device_capability()
    if major < 10 or major == 12:
        raise RuntimeError(
            f"wo_a_mega requires SM100+ excluding SM12x; got SM{major}{minor}"
        )
    return load_jit(
        f"wo_a_mega_m{max_tokens}",
        cuda_files=["deepseek_v4/wo_a_mega.cuh"],
        cuda_wrappers=[("run", f"wo_a_mega_run<{max_tokens}>")],
        extra_cuda_cflags=["-O3"],
        # cuTensorMapEncodeTiled is a driver-API call; the stub resolves at link
        # time and the real libcuda.so.1 comes from the driver at runtime.
        extra_ldflags=[f"-L{cuda_stubs_dir()}", "-lcuda"],
    )


def wo_a_mega(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    *,
    out_mxfp8: bool = True,
) -> Sequence[torch.Tensor]:
    """Inverse-RoPE ``x`` in flight, apply ``einsum('tgd,grd->tgr')``, quantize to MXFP8.

    ``x`` is the attention output viewed as ``[T, 2, 4096]``; it may be a strided
    view of a ``[T, 64, 512]`` buffer, and the RoPE is applied to the trailing 64
    lanes of each of the 8 heads per group, exactly as ``fused_rope_inplace`` does
    -- but without writing back to ``x``.

    Returns ``(q, scales)`` when ``out_mxfp8``, else ``(y,)`` holding the
    BF16 result before quantization. The two forms are mutually exclusive:
    emitting both would make the epilogue store twice for a result no caller
    wants.
    """
    num_tokens = x.shape[0]
    assert num_tokens <= 32
    max_tokens = 16 if num_tokens <= 16 else 32
    module = _jit_module(max_tokens)
    if out_mxfp8:
        q = torch.empty((num_tokens, N_OUT), dtype=torch.float8_e4m3fn, device=x.device)
        scales = torch.empty(SCALE_BYTES, dtype=torch.uint8, device=x.device)
        module.run(x, weight, freqs_cis, positions, q, scales, None)
        return q, scales
    else:
        y = x.new_empty((num_tokens, N_OUT))
        module.run(x, weight, freqs_cis, positions, None, None, y)
        return (y,)
