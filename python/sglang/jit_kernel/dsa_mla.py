"""DeepSeek Sparse Attention (DSA) MLA decode kernel — SM100 / B200.

Implements the benchmark
  dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64
from flashinfer-bench.  Fixed hyperparameters:
  num_qo_heads = 16
  head_dim_ckv = 512  (compressed KV lora rank; serves both as K and V)
  head_dim_kpe = 64   (RoPE dim, broadcast across Q heads, only on K side)
  page_size    = 64
  topk         = 2048

Inputs:
  q_nope:  bfloat16 [num_tokens, 16, 512]
  q_pe:    bfloat16 [num_tokens, 16, 64]
  ckv_cache: bfloat16 [num_pages, 64, 512]
  kpe_cache: bfloat16 [num_pages, 64, 64]
  sparse_indices: int32 [num_tokens, 2048]   (absolute token indices = page*64 + offset; -1 = pad)
  sm_scale: float

Outputs:
  output:  bfloat16 [num_tokens, 16, 512]
  lse:     float32  [num_tokens, 16]       (log2-scaled: logsumexp(scaled_logits)/ln2)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    get_jit_cuda_arch,
    load_jit,
    override_jit_cuda_arch,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
PAGE_SIZE = 64
TOPK = 2048


# --------------------------------------------------------------------------- #
#  Reference (torch, FP32 accum) — straight transcription of the bench JSON.
# --------------------------------------------------------------------------- #


@torch.no_grad()
def ref_dsa_mla_decode(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    device = q_nope.device

    Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)

    output = torch.zeros(
        (num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    for t in range(num_tokens):
        indices = sparse_indices[t]
        valid_mask = indices != -1
        valid_indices = indices[valid_mask]
        if valid_indices.numel() == 0:
            output[t].zero_()
            continue
        tok_idx = valid_indices.to(torch.long)

        Kc = Kc_all[tok_idx]
        Kp = Kp_all[tok_idx]
        qn = q_nope[t].to(torch.float32)
        qp = q_pe[t].to(torch.float32)

        logits = (qn @ Kc.T) + (qp @ Kp.T)
        logits_scaled = logits * sm_scale

        lse[t] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        attn = torch.softmax(logits_scaled, dim=-1)
        out = attn @ Kc
        output[t] = out.to(torch.bfloat16)

    return output, lse


# --------------------------------------------------------------------------- #
#  JIT CUDA kernel — loaded on first call.
# --------------------------------------------------------------------------- #


@cache_once
def _jit_module() -> "Module":
    cuda_arch = get_jit_cuda_arch()
    assert (cuda_arch.major, cuda_arch.minor) == (
        10,
        0,
    ), "dsa_mla_decode requires SM100 (B200)"
    with override_jit_cuda_arch(10, 0, "a"):
        return load_jit(
            "dsa_mla_decode",
            cuda_files=["dsa_mla/dsa_mla_decode.cuh"],
            cuda_wrappers=[("dsa_mla_decode", "dsa_mla_decode")],
            extra_cuda_cflags=["--use_fast_math"],
        )


def dsa_mla_decode(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: float,
    output: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the sparse MLA decode kernel.

    Shapes are fixed by benchmark spec (h16/ckv512/kpe64/topk2048/ps64).
    """
    num_tokens = q_nope.size(0)
    device = q_nope.device
    if output is None:
        output = torch.empty(
            (num_tokens, NUM_QO_HEADS, HEAD_DIM_CKV),
            dtype=torch.bfloat16,
            device=device,
        )
    if lse is None:
        lse = torch.empty(
            (num_tokens, NUM_QO_HEADS), dtype=torch.float32, device=device
        )
    module = _jit_module()
    module.dsa_mla_decode(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        output,
        lse,
        float(sm_scale),
    )
    return output, lse
