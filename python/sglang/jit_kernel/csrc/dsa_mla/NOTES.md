# DSA MLA decode — design notes (h16/ckv512/kpe64/topk2048/ps64)

Target: flashinfer-bench `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`.

## Current performance (B200, seqlen=8192)

| num_tokens | ours (µs) | flashinfer_trtllm (µs) | speedup |
|-----------:|----------:|-----------------------:|---------|
|          1 |        57 |                     80 | **1.41×** |
|          2 |        57 |                     77 | **1.35×** |
|          4 |        57 |                     79 | **1.38×** |
|          6 |        87 |                     78 |   0.90× |
|          8 |        88 |                     83 |   0.95× |

Primary regime (batch 1–4) is ~35% faster than flashinfer_trtllm. At bs≥6 we fall behind
because each CTA has to process 2 KV blocks and we don't pipeline across blocks.

## Design summary (V2, current)

- **Grid**: `(num_tokens × num_splits)` where `num_splits` is the largest divisor of
  `NUM_KV_BLOCKS=32` that fits `≤ num_sms/num_tokens`.
- **Block**: 128 threads = 4 warps.
- **Tiles**: `B_H=16` (matches num_qo_heads exactly — no pad), `B_TOPK=64`.
- **Per-CTA smem (~98 KB)**:
  - sQ: `[16, 576+8]` bf16 (row-pad to break bank conflicts at stride 576)
  - sK: `[64, 576+8]` bf16 (gather target)
  - sS: `[16, 64]` fp32 (QK^T, masked softmax logits)
  - sP: `[16, 64]` bf16 (softmaxed values for PV)
- **Loader**: per-thread `cp.async.cg.shared.global.L2::256B` 16 B loads; FA4-style.
- **Compute**: naive FMA loops (NO MMA yet), with 8 independent accumulators in QK
  for ILP (critical fix — breaks the long serial dependency chain).
- **Softmax**: one warp does all 16 rows × 64 cols per block (trivially fast).
- **O**: FP32 register tile `[16][4]` per thread; each warp owns 128 d_v cols.
- **Combine**: separate kernel that merges `(lse_s, o_s)` across splits in log2 space.

## Key learnings from the optimization path

1. **`cudaMallocAsync` per call was costing ~100 µs.** Moved to a cached workspace
   (`_WORKSPACE_CACHE` on the Python side, carved into o_accum + lse_accum).
2. **The biggest single speedup came from removing the serial FMA dependency in QK**.
   8 independent accumulators cut warp cycles/issue from 4.79 → 2.38. (+2×)
3. **Excessive `#pragma unroll` bloats the binary** and causes i-cache stalls (48% of
   warp cycles). Trimming `unroll` on the outer `c`/`r` loops dropped stalls.
4. **Split-KV is essential.** At bs=1, 1 CTA/token uses 1/148 of B200. With 32 splits
   we run at ~20% SM utilization — memory gather is not BW-bound anyway.

## Things we didn't do but would help

### (A) Tensor core (UMMA / mma.m16n8k16)

Single biggest unexplored win. Current compute budget per B_TOPK block:

- QK: 16 × 64 × 576 = 589 824 scalar FMAs
- PV: 16 × 512 × 64 = 524 288 scalar FMAs

Total ≈ 1.1 M FMAs per block. With `mma.m16n8k16` (M=16 fits exactly — no padding
needed for h_q=16), this drops to ≈ 32 UMMA calls per block. Expected speedup: 5–10×
on the compute part, which is currently ~50% of kernel time (the rest is HBM gather
latency).

Attempted in an early commit but had an ldmatrix→mma register mapping bug; reverted
to keep correctness progress. Re-introducing it requires a careful isolated test of
each 16×16 fragment load before plumbing in.

### (B) Double-buffered K loads

At bs=6,8 each CTA processes 2 KV blocks sequentially → each load serialises with its
compute. Double-buffering (167 KB smem) would hide one load behind compute, cutting
the bs=8 time from 88 → ~65 µs. Attempt regressed at bs=1 (no multi-iter to overlap);
needs a guard to only enable when `blk_end - blk_start >= 2`.

### (C) 2-CTA cluster cooperative gather

At bs=1, 32 CTAs on 148 SMs fit in ~20% of the GPU. Pairing CTAs into 2-CTA clusters
via `cluster_shape=(2,1,1)` and splitting the 64-token K block between them (each CTA
loads 32 tokens, then swaps halves via DSMEM) could double effective HBM throughput
per token. Would need `cp.async.bulk.shared::cluster` plumbing.

### (D) Sort indices by HBM row buffer

Sparse gather of 64 × 576 B from random positions trashes HBM row locality. A
5-bit-bucket radix sort of the 64 indices in the indices-xform step (FA4 / FlashMLA
both skip this) should lift gather throughput 20–40% when page misses dominate.

### (E) Batch-pack query tokens with shared indices

When MTP q_len_per_req > 1 or draft-decode tokens share the same top-k list (common in
speculative decoding), pack up to 4 queries into `M=64` and attend once. Pays back the
MMA oversize for h_q=16.

## Files

- `dsa_mla_decode.cuh` — kernel (single file, uses sgl_kernel JIT headers).
- `../../dsa_mla.py` — Python wrapper + torch reference.
- `../../tests/dsa_mla/test_dsa_mla_decode.py` — 10 correctness tests (pass).
- `../../benchmark/dsa_mla/bench_dsa_mla_decode.py` — compares against flashinfer.
