"""LeapQuant GDN decode (T = 1) for Qwen3.5 GDN layers (H_k 16, H_v 32, K = V = 128).

One launch per layer replaces production's unpack + conv1d update, packed recurrence and gated RMSNorm.
State format: ``gdn_lq_format.cuh`` (aliases the fp32 temporal slot ``[S, 32, 128, 128]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

REFIT_NONE = 0  # write l + 1; the caller refits when l >= lim
REFIT_WRAP = 1  # timing only: l wraps to 0 at lim without refitting
REFIT_SHELL = 2  # timing only: same loads and stores, no math
REFIT_FUSED = 3  # the CTA whose window fills refits both value heads in place (cold1)
REFIT_LIST = 4  # append filled (request, key head) units to `due`; gdn_lq_decode_flush refits them (PDL)


@cache_once
def _jit_persistent_module(groups: int, stages: int, debug: bool, refit: int) -> Module:
    args = make_cpp_args(groups, stages, debug, refit)
    return load_jit(
        "gdn_lq_decode_persistent",
        *args,
        cuda_files=["attention/gdn_lq_decode.cuh"],
        cuda_wrappers=[("run", f"GdnLqDecodePersistent<{args}>::run")],
    )


@cache_once
def _jit_flush_module() -> Module:
    return load_jit(
        "gdn_lq_decode_flush",
        cuda_files=["attention/gdn_lq_decode.cuh"],
        cuda_wrappers=[("run", "GdnLqDecodeFlush::run")],
    )


def make_due_list(num_requests: int, device) -> torch.Tensor:
    """Workspace of REFIT_LIST / gdn_lq_decode_flush: count | done | units; zero once, the flush resets it."""
    return torch.zeros(2 + 16 * num_requests, dtype=torch.int32, device=device)


def gdn_lq_decode_flush(pool, cache_indices, due, q0, pdl: bool = True) -> None:
    """Refit every (request, key head) the preceding REFIT_LIST step appended to `due` (launched with PDL)."""
    _jit_flush_module().run(pool, cache_indices, due, q0, pdl)


@cache_once
def _jit_decode_module(debug: bool, refit: int) -> Module:
    args = make_cpp_args(debug, refit)
    return load_jit(
        "gdn_lq_decode",
        *args,
        cuda_files=["attention/gdn_lq_decode.cuh"],
        cuda_wrappers=[("run", f"GdnLqDecode<{args}>::run")],
    )


_EMPTY = {}


def _empty(device):
    t = _EMPTY.get(device)
    if t is None:
        t = _EMPTY[device] = torch.empty(0, dtype=torch.float32, device=device)
    return t


def gdn_lq_decode(
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    cache_indices: torch.Tensor,
    pool: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    out: Optional[torch.Tensor] = None,
    refit: int = REFIT_NONE,
    dbg: Optional[torch.Tensor] = None,
    persistent: Optional[tuple] = None,
    q0: Optional[torch.Tensor] = None,
    due: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """qkvz [N, 12288] bf16 (in_proj output), ba [N, 64] bf16, cache_indices [N] int32,
    pool [S, 32, 128, 128] fp32 (one layer of the temporal pool, LeapQuant format), conv_state [S, 8192, 3] bf16,
    conv_weight [8192, 4] bf16, A_log [32] fp32, dt_bias [32] bf16, norm_weight [128] bf16.
    Returns the normed output [N, 4096] bf16. dbg (optional, [N, 8192 + 4096] fp32): conv output | pre-norm o."""
    if out is None:
        out = qkvz.new_empty(qkvz.shape[0], 4096)
    if persistent is not None:  # (consumer groups, stages)
        mod = _jit_persistent_module(
            persistent[0], persistent[1], dbg is not None, refit
        )
    else:
        mod = _jit_decode_module(dbg is not None, refit)
    mod.run(
        qkvz,
        ba,
        cache_indices,
        pool,
        conv_state,
        conv_weight,
        A_log,
        dt_bias,
        norm_weight,
        out,
        dbg if dbg is not None else _empty(qkvz.device),
        eps,
        due if refit == REFIT_LIST else (q0 if q0 is not None else _empty(qkvz.device)),
    )
    return out
