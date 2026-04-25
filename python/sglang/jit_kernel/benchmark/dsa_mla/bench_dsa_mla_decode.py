"""Benchmark sparse MLA decode — compares jit kernel vs the official flashinfer
wrapper at baselines/dsa_sparse_attention/main.py.

Both runners use the SAME public signature
    run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale) -> (output,)
and ALL preprocessing (seq_lens derivation, layout transforms, workspace alloc)
runs INSIDE that call. The timing window is therefore an honest "what does
deployment cost" comparison.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Callable

import torch

from sglang.jit_kernel.dsa_mla import (
    HEAD_DIM_CKV,
    HEAD_DIM_KPE,
    NUM_QO_HEADS,
    PAGE_SIZE,
    TOPK,
    dsa_mla_decode,
)


_OFFICIAL_PATH = "/data/dark/Ave-Mujica/.vscode/baselines/dsa_sparse_attention"


def _load_official_run() -> Callable:
    if _OFFICIAL_PATH not in sys.path:
        sys.path.insert(0, _OFFICIAL_PATH)
    import importlib

    main_mod = importlib.import_module("main")
    return main_mod.run


def make_inputs(num_tokens: int, seqlen: int, seed: int = 0, device: str = "cuda"):
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    num_pages = (seqlen + PAGE_SIZE - 1) // PAGE_SIZE
    q_nope = torch.randn((num_tokens, NUM_QO_HEADS, HEAD_DIM_CKV), dtype=torch.bfloat16, device=device)
    q_pe = torch.randn((num_tokens, NUM_QO_HEADS, HEAD_DIM_KPE), dtype=torch.bfloat16, device=device)
    ckv = torch.randn((num_pages, PAGE_SIZE, HEAD_DIM_CKV), dtype=torch.bfloat16, device=device)
    kpe = torch.randn((num_pages, PAGE_SIZE, HEAD_DIM_KPE), dtype=torch.bfloat16, device=device)
    indices = torch.empty((num_tokens, TOPK), dtype=torch.int32, device=device)
    for t in range(num_tokens):
        k = min(TOPK, seqlen)
        perm = torch.randperm(seqlen, generator=g)[:k].to(torch.int32).to(device)
        if k < TOPK:
            pad = torch.full((TOPK - k,), -1, dtype=torch.int32, device=device)
            indices[t] = torch.cat([perm, pad])
        else:
            indices[t] = perm
    sm_scale = 1.0 / math.sqrt(192)
    return q_nope, q_pe, ckv, kpe, indices, sm_scale


def bench(fn: Callable, args, warmup: int = 20, rep: int = 200) -> float:
    """Return median latency in microseconds."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(rep):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        fn(*args)
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1))  # ms
    times.sort()
    return times[len(times) // 2] * 1000.0  # µs


def run_ours(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    return dsa_mla_decode(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)


# ---- flash_mla baseline ----
# FlashMLA's SM100 sparse decode requires h_q to be a multiple of 128. For h=16
# we pad q to 128 heads (the other 112 are dummy), let the kernel run on h=128,
# then slice the first 16 heads back. This is the canonical way to call
# flash_mla_sparse_fwd from a model with smaller h_q (see
# sglang/srt/layers/attention/nsa_backend.py:_forward_flashmla_sparse).
_FLASH_MLA_PAD = 128  # Blackwell minimum


def run_flash_mla(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd
    T, H, _ = q_nope.shape
    num_pages, page_size, _ = ckv_cache.shape
    D_QK = HEAD_DIM_CKV + HEAD_DIM_KPE
    # Pad q to [T, 128, D_QK].
    q_padded = q_nope.new_zeros((T, _FLASH_MLA_PAD, D_QK))
    q_padded[:, :H, :HEAD_DIM_CKV] = q_nope
    q_padded[:, :H, HEAD_DIM_CKV:] = q_pe
    # Concatenate kv into [num_pages*page_size, 1, D_QK] (h_kv=1).
    kv = torch.cat([ckv_cache, kpe_cache], dim=-1).reshape(
        num_pages * page_size, 1, D_QK
    )
    # indices: [s_q, h_kv=1, topk]
    idx = sparse_indices.unsqueeze(1)
    out, _, _ = flash_mla_sparse_fwd(q_padded, kv, idx, float(sm_scale))
    return (out[:, :H, :],)  # slice back to h=16


RUNNERS = {
    "ours": run_ours,
    "flash_mla": run_flash_mla,
}


def format_bw(num_tokens: int, us: float) -> str:
    per_tok_k_bytes = TOPK * (HEAD_DIM_CKV * 2 + HEAD_DIM_KPE * 2)
    total_bytes = num_tokens * per_tok_k_bytes
    return f"{total_bytes / (us * 1e-6) / 1e9:.1f} GB/s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tokens", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--seqlen", nargs="+", type=int, default=[8192])
    ap.add_argument("--runners", nargs="+", default=None)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--rep", type=int, default=200)
    ap.add_argument("--no-official", action="store_true",
                    help="skip the official flashinfer wrapper baseline")
    args = ap.parse_args()

    runners = dict(RUNNERS)
    if not args.no_official:
        try:
            runners["official_flashinfer"] = _load_official_run()
        except Exception as e:
            print(f"[warn] could not load official baseline: {e}", file=sys.stderr)

    if args.runners:
        runners = {k: runners[k] for k in args.runners if k in runners}

    # Trigger JIT compile once.
    if "ours" in runners:
        run_ours(*make_inputs(1, 1024))
    torch.cuda.synchronize()

    header = f"{'tokens':>6} {'seqlen':>7} {'runner':>22} {'µs':>10} {'bw':>14}"
    print(header)
    print("-" * len(header))

    for T in args.num_tokens:
        for S in args.seqlen:
            inputs = make_inputs(T, S)
            for name, fn in runners.items():
                try:
                    us = bench(fn, inputs, warmup=args.warmup, rep=args.rep)
                    print(f"{T:>6} {S:>7} {name:>22} {us:>10.2f} {format_bw(T, us):>14}")
                except Exception as e:
                    print(f"{T:>6} {S:>7} {name:>22}  ERROR: {e}")


if __name__ == "__main__":
    main()
