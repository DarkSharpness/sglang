"""LeapQuant 8-bit GDN state: refit core, pack / unpack, commit (JIT CUDA). Experimental."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Format constants (csrc/attention/gdn_lq_format.cuh)
DIM = 128
RANK = 4
C_DEFAULT = 20


def layout(C: int = C_DEFAULT) -> dict:
    off = {}
    o = 0
    for name, nbytes in (
        ("Z", DIM * DIM),
        ("s", DIM * 4),
        ("c", DIM * 4),
        ("Kt", RANK * DIM * 2),
        ("Ut", RANK * DIM * 2),
        ("u", C * DIM * 2),
        ("kh", C * DIM * 2),
        ("G", (C * 4 + 15) // 16 * 16),
    ):
        off[name] = o
        o += nbytes
    off["l"] = o
    off["lim"] = o + 4
    off["bytes"] = o + 16
    return off


@cache_once
def _refit_module(C: int, iters: int, warm: bool, profile: bool = False) -> Module:
    import os

    args = make_cpp_args(C, iters, warm)
    flags = ["-DLQR_PROFILE"] if profile else []
    exp = os.environ.get("LQR_EXP", "")  # experiment-time macro variants, e.g. "ORTH32"
    flags += [f"-DLQR_{x}" for x in exp.split(",") if x]
    return load_jit(
        "gdn_lq_refit"
        + ("_prof" if profile else "")
        + "".join(f"_{x.lower()}" for x in exp.split(",") if x),
        *args,
        cuda_files=["attention/gdn_lq_refit.cuh"],
        cuda_wrappers=[
            ("refit_list", f"gdn_lq_refit_list<{args}>"),
            ("commit", f"gdn_lq_commit<{args}>"),
        ],
        extra_cuda_cflags=flags or None,
    )


def refit_list(
    pool: torch.Tensor,
    layer: torch.Tensor,
    slot: torch.Tensor,
    lnew: torch.Tensor,
    q0: torch.Tensor,
    C: int = C_DEFAULT,
    iters: int = 2,
    warm: bool = True,
    floor: float = 1e-6,
    stamps: torch.Tensor | None = None,
) -> None:
    """Refit every value head of the listed (layer, slot) items in place (test entry). stamps: int64
    [items * HV, 24, 8] clock64 per phase and warp (profile build)."""
    mod = _refit_module(C, iters, warm, stamps is not None)
    mod.refit_list(
        pool, layer, slot, lnew, q0, floor, 0 if stamps is None else stamps.data_ptr()
    )


def commit(
    pool: torch.Tensor,
    slot_idx: torch.Tensor,
    n_acc: torch.Tensor | None,
    q0: torch.Tensor,
    P: int = 16,
    C: int = C_DEFAULT,
    iters: int = 1,
    warm: bool = False,
    floor: float = 1e-6,
    ws: torch.Tensor | None = None,
    track_idx: torch.Tensor | None = None,
    track_steps: torch.Tensor | None = None,
) -> None:
    """Step commit over every layer of pool [L, S, HV, 128, 128] (format view): l += n_acc (None: decode flush, n = 0),
    refit every value head of the (layer, request) pairs with l >= lim, write l / lim to every value-head copy.
    Tracking (extra_buffer prefix cache): requests with track_idx >= 0 and track_steps >= 0 get their slot's format
    copied into track_idx with l = l_old + track_steps + 1, taken before the refit (int32 [N] each).
    One scan launch (+ one track-copy launch when tracking) + one persistent refit launch; graph-safe."""
    L, N = pool.shape[0], slot_idx.shape[0]
    if ws is None:
        ws = torch.empty(6 * L * N + 2, dtype=torch.int32, device=pool.device)
    none = slot_idx[:0]
    _refit_module(C, iters, warm).commit(
        pool,
        slot_idx,
        none if n_acc is None else n_acc,
        q0,
        ws,
        none if track_idx is None else track_idx,
        none if track_steps is None else track_steps,
        P,
        floor,
    )


@cache_once
def _verify_module(
    C: int, shell: bool = False, hpc: int = 2, min_blocks: int = 2
) -> Module:
    import os

    args = make_cpp_args(C, shell, hpc, min_blocks)
    exp = [
        x for x in os.environ.get("LQV_EXP", "").split(",") if x
    ]  # experiment-time macros, e.g. MINB=3
    return load_jit(
        "gdn_lq_verify" + "".join(f"_{x.lower().replace('=', '')}" for x in exp),
        *args,
        extra_cuda_cflags=[f"-DLQV_{x}" for x in exp] or None,
        cuda_files=["attention/gdn_lq_verify.cuh"],
        cuda_wrappers=[("verify", f"gdn_lq_verify<{args}>")],
    )


def verify(
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    slot_idx: torch.Tensor,
    row_idx: torch.Tensor,
    pool_layer: torch.Tensor,
    conv_state: torch.Tensor,
    conv_w: torch.Tensor,
    conv_win: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_w: torch.Tensor,
    eps: float,
    out: torch.Tensor | None = None,
    C: int = C_DEFAULT,
    shell: bool = False,
    hpc: int | None = None,
) -> torch.Tensor:
    """Fused LeapQuant GDN target verify (T = 4) of one layer. qkvz [N*4, 12288], ba [N*4, 64] bf16; pool_layer
    [S, 32, 128, 128] fp32 (format view); conv_state [S, 8192, 3], conv_w [8192, 4], conv_win [rows, 8192, 6] bf16;
    A_log, dt_bias [32] fp32; norm_w [128] bf16. Returns the normed output [N*4, 4096] bf16."""
    if out is None:
        out = torch.empty(qkvz.shape[0], 4096, dtype=qkvz.dtype, device=qkvz.device)
    if hpc is None:
        import os

        hpc = int(
            os.environ.get("LQV_HPC", "2")
        )  # value heads per CTA (1 or 2); experiment knob
    # register budget: 3 CTAs/SM (85 regs) pays off once the grid has several waves (measured: N=48 27.2 vs 30.0 us,
    # N <= 16 slower from spills)
    n_req = slot_idx.shape[0]
    min_blocks = (3 if n_req >= 32 else 2) if hpc == 2 else (6 if n_req >= 32 else 4)
    _verify_module(C, shell, hpc, min_blocks).verify(
        qkvz,
        ba,
        slot_idx,
        row_idx,
        pool_layer,
        conv_state,
        conv_w,
        conv_win,
        A_log,
        dt_bias,
        norm_w,
        out,
        eps,
    )
    return out
