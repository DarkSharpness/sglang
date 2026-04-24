"""Benchmark sparse MLA decode — compares jit kernel vs flashinfer/FA4 baselines.

Spec: dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Callable, Dict

import torch

from sglang.jit_kernel.dsa_mla import (
    HEAD_DIM_CKV,
    HEAD_DIM_KPE,
    NUM_QO_HEADS,
    PAGE_SIZE,
    TOPK,
    dsa_mla_decode,
    ref_dsa_mla_decode,
)


def make_inputs(num_tokens: int, seqlen: int, seed: int = 0, device: str = "cuda"):
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    num_pages = (seqlen + PAGE_SIZE - 1) // PAGE_SIZE
    q_nope = torch.randn((num_tokens, NUM_QO_HEADS, HEAD_DIM_CKV), dtype=torch.bfloat16, device=device)
    q_pe = torch.randn((num_tokens, NUM_QO_HEADS, HEAD_DIM_KPE), dtype=torch.bfloat16, device=device)
    ckv = torch.randn((num_pages, PAGE_SIZE, HEAD_DIM_CKV), dtype=torch.bfloat16, device=device)
    kpe = torch.randn((num_pages, PAGE_SIZE, HEAD_DIM_KPE), dtype=torch.bfloat16, device=device)
    indices = torch.empty((num_tokens, TOPK), dtype=torch.int32, device=device)
    total = num_pages * PAGE_SIZE
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


def bench(fn: Callable, args, warmup: int = 10, rep: int = 100) -> float:
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


# --- Candidate runners ---

def run_ours(q_nope, q_pe, ckv, kpe, idx, scale):
    out, lse = dsa_mla_decode(q_nope, q_pe, ckv, kpe, idx, scale)
    return out, lse


def run_flashinfer_trtllm(q_nope, q_pe, ckv, kpe, idx, scale):
    """Uses flashinfer's trtllm_batch_decode_with_kv_cache_mla.

    Needs block_tables + seq_lens form.  This is the API name the bench targets.
    """
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla  # type: ignore

    # Construct merged kv_cache of shape [num_pages, 1, page_size, D_CKV + D_KPE]
    # (3-D is also accepted; use 4-D for newer flashinfer.)
    num_pages, page_size, _ = ckv.shape
    kv_cat = torch.cat([ckv, kpe], dim=-1)  # [P, 64, 576]
    kv_cache = kv_cat.unsqueeze(1)  # [P, 1, 64, 576]

    # Build query in the expected layout: [B*S, H, d_qk] concatenated nope||pe.
    q_cat = torch.cat([q_nope, q_pe], dim=-1)  # [T, 16, 576]
    query = q_cat.unsqueeze(1)  # [T, 1, 16, 576] (q_len_per_request=1)

    num_tokens = q_nope.size(0)
    # block_tables for sparse MLA: shape [batch, q_len_per_req=1, topk].
    block_tables = idx.unsqueeze(1)  # [T, 1, 2048]

    seq_lens = torch.full((num_tokens,), TOPK, dtype=torch.int32, device=q_nope.device)
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=q_nope.device)

    out = trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        qk_nope_head_dim=HEAD_DIM_CKV,  # note: misnamed; pass kv_lora_rank here
        kv_lora_rank=HEAD_DIM_CKV,
        qk_rope_head_dim=HEAD_DIM_KPE,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=TOPK,
        sparse_mla_top_k=TOPK,
        bmm1_scale=float(scale),
        bmm2_scale=1.0,
    )
    return out


def run_flash_attention_v4(q_nope, q_pe, ckv, kpe, idx, scale):
    """Uses FA4 cute-dsl decode path.  Best-effort; may require padding q heads."""
    # Padding h=16 → h=128 by replication is the FA4 DSA assumption (MQA 128).
    # For a quick benchmark, use flash_attn_varlen_func with top-k indices ... skipped.
    raise NotImplementedError


BENCHES: Dict[str, Callable] = {
    "ours": run_ours,
    "flashinfer_trtllm": run_flashinfer_trtllm,
}


def format_bw(seqlen: int, num_tokens: int, us: float) -> str:
    # Approx bytes moved per token: topk * (ckv*2 + kpe*2) + q/output
    per_tok_k_bytes = TOPK * (HEAD_DIM_CKV * 2 + HEAD_DIM_KPE * 2)
    total_bytes = num_tokens * per_tok_k_bytes
    gb_per_s = total_bytes / (us * 1e-6) / 1e9
    return f"{gb_per_s:.1f} GB/s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tokens", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--seqlen", nargs="+", type=int, default=[4096, 8192, 16384])
    ap.add_argument("--runners", nargs="+", default=None)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rep", type=int, default=50)
    args = ap.parse_args()

    runners = args.runners or list(BENCHES.keys())

    # Trigger JIT compile once.
    if "ours" in runners:
        inp = make_inputs(1, 1024)
        run_ours(*inp)
    torch.cuda.synchronize()

    header = f"{'tokens':>6} {'seqlen':>7} {'runner':>22} {'µs':>10} {'bw':>14}"
    print(header)
    print("-" * len(header))

    for T in args.num_tokens:
        for S in args.seqlen:
            inp = make_inputs(T, S)
            for name in runners:
                try:
                    us = bench(BENCHES[name], inp, warmup=args.warmup, rep=args.rep)
                    bw = format_bw(S, T, us)
                    print(f"{T:>6} {S:>7} {name:>22} {us:>10.2f} {bw:>14}")
                except Exception as e:
                    print(f"{T:>6} {S:>7} {name:>22}  ERROR: {e}")


if __name__ == "__main__":
    main()
