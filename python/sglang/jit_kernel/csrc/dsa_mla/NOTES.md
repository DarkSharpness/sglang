# DSA MLA decode — design notes (h16/ckv512/kpe64/topk2048/ps64)

Target: flashinfer-bench `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`.

## Performance reference (B200, seqlen=8192)

Apples-to-apples comparison. All three runners share the public signature
`run(q_nope, q_pe, ckv, kpe, idx, sm_scale) → (output,)` and ALL preprocessing
(seq_lens derivation, layout xforms, workspace alloc, q-pad-to-128 for flash_mla)
runs INSIDE the timed call. Bench source: `bench_dsa_mla_decode.py`.

### V3 (scalar FMA, last benchmarked baseline)

| num_tokens | ours (µs) | flash_mla (µs) | official flashinfer (µs) | vs flash_mla |
|-----------:|----------:|---------------:|-------------------------:|--------------|
|          1 |        55 |             74 |                       96 | **1.35×**    |
|          2 |        53 |             72 |                       97 | **1.36×**    |
|          4 |        54 |             73 |                      100 | **1.37×**    |
|          6 |        75 |             74 |                       98 |  ≈           |
|          8 |        76 |             74 |                      102 |  ≈           |
|         16 |       121 |             74 |                      102 |  0.61×       |
|         32 |       220 |             75 |                      107 |  0.34×       |

Beat flash_mla up to T=4, matched it through T=8, **scaled linearly above** because
the scalar bf16 FMA path bottlenecks on per-CTA compute as T grows.

### V4 (tensor core, current — UNTESTED on hardware)

Replaced both QK and PV inner loops with `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
(NVIDIA mma instruction, available since SM80; on SM100/B200 this dispatches to
the same tensor-core pipeline FlashMLA's UMMA goes through).

Expected per-block compute drop:
- QK: 590K scalar FMAs → ~72 mma calls/warp ≈ 0.3 µs/warp (was ~9 µs).
- PV: 524K scalar FMAs → ~64 mma calls/warp ≈ 0.25 µs/warp (was ~10 µs).

Per block compute drops ~30× and the kernel becomes HBM-gather-bound. Expected
T=32 wall-clock should approach ~80–100 µs (vs 220 µs scalar). Bench numbers
will be filled in once the test machine is back; the code currently compiles
clean against `mma.m16n8k16` semantics but has not been run on hardware.

The flash_mla baseline pads `q` from 16 → 128 heads (Blackwell minimum supported
by `flash_mla_sparse_fwd`) and slices the output back; this is exactly how
sglang's `nsa_backend._forward_flashmla_sparse` invokes it for h_q < 128.

## Design summary (V4)

- **Static split-KV scheduler.** `num_splits = largest divisor of NUM_KV_BLOCKS=32`
  that fits `num_tokens × num_splits ≤ 2 × num_sms`. Targets 2 CTAs/SM
  occupancy (`__launch_bounds__(NUM_THREADS, 2)`) for cross-CTA load latency
  hiding. Host-computed; **no preprocessing kernel**.
- **Compute**: `mma.m16n8k16.row.col.f32.bf16.bf16.f32` for both QK and PV.
  M=16 fits exactly to `B_H=16` so no Q-padding is needed.
  - QK: per warp does 2 N-sub-tiles × 36 K-tiles = 72 MMA calls. Both A (sQ)
    and B (sK transposed-via-row-major) load with simple `.b32` smem reads
    since sK[n][k] naturally has K-contiguous-per-N layout matching mma.B "col".
  - PV: per warp does 16 N-sub-tiles × 4 K-tiles = 64 MMA calls. A (sP) is
    contiguous; B (V = sK[..., 0:512]) needs strided bf16 reads because sV
    is K-major in our smem but mma.B wants col-layout. Each thread does 4
    strided 16-bit loads per K-iter, packed into 2 .b32. Less efficient than
    `ldmatrix.trans` (TODO) but correct without CUTLASS layouts.
  - Per-thread accumulator `rO[16][4]`: 16 N-sub-tiles × 4 fp32-fragments,
    same total registers as the previous scalar `[B_H][4]` (64 fp32/thread).
- **Block**: 128 threads = 4 warps. `B_H = 16`, `B_TOPK = 64`.
- **Per-CTA smem (~98 KB)**: sQ + sK + sS + sP, single buffer.
- **Loader**: per-thread `cp.async.cg.shared.global.L2::256B`.
- **Compute**: scalar FMA loops with 8 independent accumulators (ILP win); no MMA yet.
- **Combine kernel**: templated on `NUM_SPLITS ∈ {2,4,8,16,32}`. Issues all
  NUM_SPLITS o_accum loads back-to-back into a register array (memory-level
  parallelism), then a pure-compute pass does softmax + accumulate.
- **lse**: optional (signal "skip" by passing a 0-numel tensor); the Python API
  exposes `dsa_mla_decode_with_lse` for tests.

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

### (A) Tensor core (mma.m16n8k16) — IMPLEMENTED in V4

Both QK and PV inner loops now use `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
(see `mma_m16n8k16_bf16_f32` helper). M=16 fits exactly to `B_H=16`, so no padding.

QK is the clean case: sQ row-major fits mma.A "row"; sK[n][k] (K-major) naturally
matches mma.B "col" since fixed-n + varying-k is contiguous in our smem.

PV is awkward: V is sK[..., 0:512] with K-major layout, but mma.B "col" wants
N-outer in memory. The current implementation does 4 strided bf16 reads per
thread per (k_iter, n_sub) and packs into 2 .b32. This works but leaves perf
on the table due to bank contention. Follow-up: use `ldmatrix.x2.trans` to do
the transposed load in hardware (FlashMLA's path via `SmemLayoutKTilesTransposed_SW128`).

**This change has not been tested on real hardware** — the dev box was down at
the time of writing. The next session should:
1. Run the existing test suite (`tests/dsa_mla/test_dsa_mla_decode.py`).
2. Run `bench_dsa_mla_decode.py` and compare against V3 numbers above.
3. If correctness fails, inspect the per-thread mma fragment layout (the
   per-thread layout comments next to `mma_m16n8k16_bf16_f32` are the spec).

### (B) Double-buffered K loads

Tried with NUM_BUFS=2 (FlashMLA-style ring buffer): the 2× K smem (146 KB) blows
past the 2-CTA/SM budget so we drop to 1 CTA/SM — which loses the cross-CTA load
hiding that the 2-CTA/SM packing currently buys (~10 µs at low T, ~50 µs at T=8).
Net regression of 13–22 µs across the board. The ~15 µs we'd save by hiding K
load behind compute is less than what we lose to halved occupancy.

To make this win, would need to reduce K smem (smaller B_TOPK, or split K into
NoPE/RoPE halves and only double-buffer NoPE). Or use cluster mode (C) to share
the buffer between 2 CTAs.

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
