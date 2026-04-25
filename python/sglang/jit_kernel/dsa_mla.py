"""DeepSeek Sparse Attention (DSA) MLA decode kernel — SM100 / B200.

Implements the benchmark
  dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64
from flashinfer-bench.  Fixed hyperparameters:
  num_qo_heads = 16
  head_dim_ckv = 512  (compressed KV lora rank; serves both as K and V)
  head_dim_kpe = 64   (RoPE dim, broadcast across Q heads, only on K side)
  page_size    = 64
  topk         = 2048

Public interface matches the official baseline at
  baselines/dsa_sparse_attention/main.py:
    run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale) -> (output,)
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
_NUM_KV_BLOCKS = TOPK // 64  # B_TOPK = 64


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


# Cached per-device {workspace, output, lse, empty_lse}. Workspace size depends
# on num_tokens, so we grow it lazily.
_CACHE: dict = {}
# Empty 1-D float tensor used to signal "skip lse" to the C++ shim
# (it inspects numel() == 0).
_EMPTY_LSE: dict = {}


def _bytes_for(num_tokens: int) -> int:
    # Worst-case workspace size in bytes. Mirrors the C++-side layout:
    #   o_accum             = max_total_splits * H * D_V * 4
    #   lse_accum           = max_total_splits * H * 4
    #   sched_meta          = num_sm_parts * 32  (≤ 256 SMs)
    #   num_splits_prefix   = (T+1) * 4
    #   num_blocks_per_req  = T * 4
    # Rounded up generously.
    max_total = num_tokens * _NUM_KV_BLOCKS
    o_accum = max_total * NUM_QO_HEADS * HEAD_DIM_CKV * 4
    lse_accum = max_total * NUM_QO_HEADS * 4
    misc = 256 * 32 + (num_tokens + 1) * 4 + num_tokens * 4
    # Per-stage 256-byte alignment padding.
    return o_accum + lse_accum + misc + 4096


def _get_workspace(device: torch.device, num_tokens: int) -> torch.Tensor:
    key = (str(device), "ws")
    cur = _CACHE.get(key)
    need = _bytes_for(num_tokens)
    if cur is None or cur.numel() < need:
        cur = torch.empty((need,), dtype=torch.uint8, device=device)
        _CACHE[key] = cur
    return cur


def _get_empty_lse(device: torch.device) -> torch.Tensor:
    key = str(device)
    t = _EMPTY_LSE.get(key)
    if t is None:
        t = torch.empty((0,), dtype=torch.float32, device=device)
        _EMPTY_LSE[key] = t
    return t


# --------------------------------------------------------------------------- #
#  Public API — matches `baselines/dsa_sparse_attention/main.py:run`.
# --------------------------------------------------------------------------- #


def dsa_mla_decode(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: float,
) -> Tuple[torch.Tensor]:
    """Sparse MLA decode — official wrapper signature.

    Returns `(output,)` with `output: bfloat16 [num_tokens, num_qo_heads, head_dim_ckv]`.
    All preprocessing (seq_lens derivation, scheduler metadata) happens GPU-side
    inside a single fused metadata kernel — no CPU sync.
    """
    num_tokens = q_nope.size(0)
    device = q_nope.device
    output = torch.empty(
        (num_tokens, NUM_QO_HEADS, HEAD_DIM_CKV),
        dtype=torch.bfloat16,
        device=device,
    )
    workspace = _get_workspace(device, num_tokens)
    empty_lse = _get_empty_lse(device)
    module = _jit_module()
    module.dsa_mla_decode(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        output,
        empty_lse,
        workspace,
        float(sm_scale),
    )
    return (output,)


def dsa_mla_decode_with_lse(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Test-only variant: also returns lse [num_tokens, num_qo_heads] in log2 space."""
    num_tokens = q_nope.size(0)
    device = q_nope.device
    output = torch.empty(
        (num_tokens, NUM_QO_HEADS, HEAD_DIM_CKV),
        dtype=torch.bfloat16,
        device=device,
    )
    lse = torch.empty(
        (num_tokens, NUM_QO_HEADS), dtype=torch.float32, device=device
    )
    workspace = _get_workspace(device, num_tokens)
    module = _jit_module()
    module.dsa_mla_decode(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        output,
        lse,
        workspace,
        float(sm_scale),
    )
    return output, lse
