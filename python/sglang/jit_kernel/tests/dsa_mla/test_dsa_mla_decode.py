"""Correctness tests for dsa_mla_decode JIT kernel."""

from __future__ import annotations

import math

import pytest
import torch

from sglang.jit_kernel.dsa_mla import (
    HEAD_DIM_CKV,
    HEAD_DIM_KPE,
    NUM_QO_HEADS,
    PAGE_SIZE,
    TOPK,
    dsa_mla_decode_with_lse as dsa_mla_decode,
    ref_dsa_mla_decode,
)


def _require_b200():
    if not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    cap = torch.cuda.get_device_capability(0)
    if cap[0] != 10:
        pytest.skip(f"requires sm100 (B200), got sm{cap[0]}{cap[1]}")


def _make_inputs(
    num_tokens: int,
    num_pages: int,
    seqlen: int,
    seed: int = 0,
    pad_frac: float = 0.0,
    device: str = "cuda",
):
    """Build a test case.

    Args:
        num_tokens: number of query rows (== batch size for decode)
        num_pages : allocated pages in the KV pool
        seqlen    : effective context length per token (<= num_pages * PAGE_SIZE).
                    The sparse indices are drawn from [0, seqlen).
        pad_frac  : fraction of indices set to -1 (padding)
    """
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device(device)

    q_nope = torch.randn(
        (num_tokens, NUM_QO_HEADS, HEAD_DIM_CKV), dtype=torch.bfloat16, device=dev
    )
    q_pe = torch.randn(
        (num_tokens, NUM_QO_HEADS, HEAD_DIM_KPE), dtype=torch.bfloat16, device=dev
    )
    ckv = torch.randn(
        (num_pages, PAGE_SIZE, HEAD_DIM_CKV), dtype=torch.bfloat16, device=dev
    )
    kpe = torch.randn(
        (num_pages, PAGE_SIZE, HEAD_DIM_KPE), dtype=torch.bfloat16, device=dev
    )

    total_tokens = num_pages * PAGE_SIZE
    assert seqlen <= total_tokens
    # draw TOPK indices from [0, seqlen), without replacement per token.
    indices = torch.empty((num_tokens, TOPK), dtype=torch.int32, device=dev)
    for t in range(num_tokens):
        k = min(TOPK, seqlen)
        perm = torch.randperm(seqlen, generator=g)[:k].to(torch.int32).to(dev)
        if k < TOPK:
            pad = torch.full((TOPK - k,), -1, dtype=torch.int32, device=dev)
            indices[t] = torch.cat([perm, pad])
        else:
            indices[t] = perm
    # Random additional padding
    if pad_frac > 0.0:
        mask = (torch.rand(num_tokens, TOPK, generator=g) < pad_frac).to(dev)
        indices[mask] = -1

    sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV // 4 + HEAD_DIM_KPE)  # 128+64 post-absorb
    return q_nope, q_pe, ckv, kpe, indices, sm_scale


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8])
@pytest.mark.parametrize("seqlen", [4096, 8192])
def test_output_vs_torch_ref(num_tokens, seqlen):
    _require_b200()
    num_pages = (seqlen + PAGE_SIZE - 1) // PAGE_SIZE
    q_nope, q_pe, ckv, kpe, idx, scale = _make_inputs(num_tokens, num_pages, seqlen)

    out, lse = dsa_mla_decode(q_nope, q_pe, ckv, kpe, idx, scale)
    ref_out, ref_lse = ref_dsa_mla_decode(q_nope, q_pe, ckv, kpe, idx, scale)

    out_f = out.to(torch.float32)
    ref_f = ref_out.to(torch.float32)

    abs_err = (out_f - ref_f).abs()
    rel_err = abs_err / (ref_f.abs() + 1e-6)
    print(
        f"[tokens={num_tokens} seq={seqlen}] "
        f"out  max_abs={abs_err.max():.4g} mean_abs={abs_err.mean():.4g} "
        f"max_rel={rel_err.max():.4g}"
    )
    print(
        f"[tokens={num_tokens} seq={seqlen}] "
        f"lse  max_abs={(lse - ref_lse).abs().max():.4g}"
    )

    # Tolerances from flashinfer-bench default for bf16 attention.
    assert torch.allclose(out_f, ref_f, atol=3e-2, rtol=3e-2), (
        f"output mismatch: max_abs={abs_err.max()}, max_rel={rel_err.max()}"
    )
    # LSE in log2 space — allow bigger absolute tol because of finite topk.
    assert torch.allclose(lse, ref_lse, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("pad_frac", [0.25, 0.5])
def test_output_with_padding(pad_frac):
    _require_b200()
    num_tokens = 4
    seqlen = 6000
    num_pages = (seqlen + PAGE_SIZE - 1) // PAGE_SIZE
    q_nope, q_pe, ckv, kpe, idx, scale = _make_inputs(
        num_tokens, num_pages, seqlen, pad_frac=pad_frac
    )

    out, lse = dsa_mla_decode(q_nope, q_pe, ckv, kpe, idx, scale)
    ref_out, ref_lse = ref_dsa_mla_decode(q_nope, q_pe, ckv, kpe, idx, scale)

    out_f = out.to(torch.float32)
    ref_f = ref_out.to(torch.float32)
    assert torch.allclose(out_f, ref_f, atol=5e-2, rtol=5e-2)
    assert torch.allclose(lse, ref_lse, atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
