"""DeepSeek V3 sparse indexer JIT kernel (SM100a / Blackwell).

Fused paged-logits kernel that computes:
  logits[b, t] = sum_h( relu(q[b,h] . K[b,t,h]) * w[b,h] )

Top-K selection via cluster-based radix kernel (4-round 8-bit radix, CTA clusters).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    get_jit_cuda_arch,
    load_jit,
    override_jit_cuda_arch,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_dpsk_module() -> Module:
    cuda_arch = get_jit_cuda_arch()
    assert (cuda_arch.major, cuda_arch.minor) == (10, 0), "DSV3 indexer requires SM100"
    with override_jit_cuda_arch(10, 0, "a"):
        return load_jit(
            "dsv3_indexer",
            cuda_files=["deepseek/dsv3_indexer.cuh"],
            extra_cuda_cflags=["--use_fast_math"],
            extra_ldflags=["-lcuda"],
        )


def can_use_dsv3_indexer() -> bool:
    try:
        _jit_dpsk_module()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Metadata computation
# ---------------------------------------------------------------------------

_SCHED_STEP = 256  # BLOCK_KV(128) * EPILOGUE_WARPGRPS(2)


def get_indexer_metadata(
    seq_lens: torch.Tensor,
    num_sms: Optional[int] = None,
) -> torch.Tensor:
    """Compute ScheduleMetadata for the 2-D (batch, kv) scheduler.

    Returns:
        metadata: [num_sms, 4] int32 CUDA tensor.
                  columns: [q_begin, k_begin, q_end, k_end]
    """
    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(
            seq_lens.device
        ).multi_processor_count
    assert isinstance(num_sms, int)

    batch_size = seq_lens.shape[0]
    sl_cpu = seq_lens.cpu().tolist()

    iters_per_batch = [math.ceil(s / _SCHED_STEP) for s in sl_cpu]
    total_iters = sum(iters_per_batch)

    prefix = [0]
    for c in iters_per_batch:
        prefix.append(prefix[-1] + c)

    def _iter_to_qk(flat_iter: int) -> Tuple[int, int]:
        lo, hi = 0, batch_size - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if prefix[mid + 1] <= flat_iter:
                lo = mid + 1
            else:
                hi = mid
        return lo, (flat_iter - prefix[lo]) * _SCHED_STEP

    meta = torch.zeros(num_sms, 4, dtype=torch.int32)
    if total_iters == 0:
        return meta.to(seq_lens.device)

    share = (total_iters + num_sms - 1) // num_sms

    begin = 0
    for s in range(num_sms):
        end = min(begin + share, total_iters)
        if begin >= end:
            continue
        qb, kb = _iter_to_qk(begin)
        qe, ke = _iter_to_qk(end)
        meta[s] = torch.tensor([qb, kb, qe, ke], dtype=torch.int32)
        begin = end

    return meta.to(seq_lens.device, non_blocking=True)


# ---------------------------------------------------------------------------
# Indexer wrapper
# ---------------------------------------------------------------------------


def dsv3_indexer(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    sm_map: Optional[torch.Tensor] = None,
    max_model_len: int = 163840,
    logits_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute fused paged logits for the DSV3 sparse indexer.

    For each batch element b and token t:
      logits[b, t] = sum_h( relu(q[b,h] . K_dequant[b,t,h]) * weights[b,h] )
    """
    batch_size = q.shape[0]
    max_model_len = ((max_model_len + 3) // 4) * 4

    if logits_out is not None:
        logits = logits_out
    else:
        logits = torch.empty(
            batch_size, max_model_len, device=q.device, dtype=torch.float32
        )

    if sm_map is None:
        sm_map = get_indexer_metadata(seq_lens)

    module = _jit_dpsk_module()
    module.dsv3_indexer(q, k_cache, weights, seq_lens, block_table, sm_map, logits)
    return logits


# ---------------------------------------------------------------------------
# Cluster-based radix top-K
# ---------------------------------------------------------------------------


# Must match TK_MAX_CHUNK in dsv3_topk_sm100.cuh
_TK_MAX_CHUNK = (227 * 1024 // 4) - (256 + 2048 + 4 + 8)  # 55796


def get_topk_num_clusters(batch_size: int, max_model_len: int) -> int:
    """Select number of CTA clusters per batch row.

    Ensures per-CTA chunk fits in smem (max ~55K elements).
    """
    # Minimum nc so per-CTA data fits in shared memory
    min_nc = (max_model_len + _TK_MAX_CHUNK - 1) // _TK_MAX_CHUNK
    for valid in (1, 2, 4, 8):
        if valid >= min_nc:
            min_nc = valid
            break
    else:
        min_nc = 8

    # Performance heuristic (B200 cluster occupancy)
    if max_model_len <= 8192:
        perf_nc = 1
    elif batch_size <= 15:
        perf_nc = 8
    elif batch_size <= 33:
        perf_nc = 4
    elif batch_size <= 74:
        perf_nc = 2
    else:
        perf_nc = 1

    return max(min_nc, perf_nc)


def dsv3_topk(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    k: int,
    num_clusters: Optional[int] = None,
    indices_out: Optional[torch.Tensor] = None,
    overflow_buf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cluster-based radix top-K on float32 logits.

    Returns:
        indices: [batch, k] int32.  -1 for padding when seq_len < k.
    """
    batch_size = logits.shape[0]
    max_model_len = logits.shape[1]
    device = logits.device

    if num_clusters is None:
        num_clusters = get_topk_num_clusters(batch_size, max_model_len)

    if indices_out is None:
        indices_out = torch.empty(batch_size, k, dtype=torch.int32, device=device)

    if overflow_buf is None:
        chunk = (max_model_len + num_clusters - 1) // num_clusters
        ov_stride = max(2048, chunk // 4)
        overflow_buf = torch.empty(
            batch_size,
            num_clusters * ov_stride * 4,
            dtype=torch.int32,
            device=device,
        )

    module = _jit_dpsk_module()
    module.dsv3_topk(logits, seq_lens, indices_out, overflow_buf, k, num_clusters)
    return indices_out


def dsv3_topk_indexer(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk: int = 2048,
    sm_map: Optional[torch.Tensor] = None,
    max_model_len: int = 163840,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused paged logits + cluster-based radix top-K selection."""
    logits = dsv3_indexer(
        q,
        k_cache,
        weights,
        seq_lens,
        block_table,
        sm_map=sm_map,
        max_model_len=max_model_len,
    )
    indices = dsv3_topk(logits, seq_lens, topk)
    return indices, logits
