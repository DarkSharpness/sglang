// DeepSeek Sparse Attention (DSA) MLA decode kernel — SM100 (B200) / h_q = 16.
//
// V1: split-KV across SMs.
//   Each CTA processes (token, split_idx) where split_idx selects a contiguous slice of the
//   32 KV blocks.  Partial (o_accum, lse_accum) are written; a combine kernel merges.
//   Grid: (num_tokens * num_splits, 1, 1)
//   num_splits = min(NUM_KV_BLOCKS, max(148 / num_tokens, 1)) rounded to divide NUM_KV_BLOCKS.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace dsa_mla_ns {

using bf16  = __nv_bfloat16;
using bf162 = __nv_bfloat162;

constexpr int NUM_HEADS     = 16;
constexpr int D_CKV         = 512;
constexpr int D_KPE         = 64;
constexpr int D_QK          = D_CKV + D_KPE;   // 576
constexpr int D_V           = 512;
constexpr int PAGE_SIZE     = 64;
constexpr int TOPK          = 2048;

constexpr int B_H           = 16;
constexpr int B_TOPK        = 64;
constexpr int NUM_WARPS     = 4;
constexpr int NUM_THREADS   = NUM_WARPS * 32;
constexpr int NUM_KV_BLOCKS = TOPK / B_TOPK;   // 32

constexpr int DV_PER_WARP   = D_V / NUM_WARPS; // 128

// ------------------ PTX helpers ------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return __cvta_generic_to_shared(const_cast<void*>(p));
}
// cp.async.cg.shared.global.L2::256B: bypass L1 → larger in-flight queue, and the .L2::256B hint
// coalesces 256-byte bursts (our row stride of 576B helps this).
__device__ __forceinline__ void cp_async_16_pred(void* smem_dst, const void* gmem_src, bool pred) {
  uint32_t s = smem_u32(smem_dst);
  if (pred) {
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" : : "r"(s), "l"(gmem_src));
  } else {
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16, 0;\n" : : "r"(s), "l"(gmem_src));
  }
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait_group() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// Tensor core MMA helper. mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32:
//   D[M=16, N=8] += A[M=16, K=16] × B[K=16, N=8]   (B in col layout)
// Per-thread fragment layout (lane t = threadIdx.x % 32):
//   A (4 .b32 / 8 bf16): rows {t/4, t/4+8}, cols {(t%4)*4..(t%4)*4+3}
//   B (2 .b32 / 4 bf16): col {t/4},          rows {(t%4)*4..(t%4)*4+3}
//   D (4 fp32):           rows {t/4, t/4+8}, cols {2*(t%4), 2*(t%4)+1}
__device__ __forceinline__ void mma_m16n8k16_bf16_f32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1));
}

// D_QK=576 means per-row byte stride = 1152 B = 288 × 4B banks.  288 % 32 = 0, so threads
// accessing the same col across rows hit the same bank — 16-way conflict in QK.
// Pad each row with 8 bf16 (16B) so stride becomes 584 bf16 = 292 banks → 292%32=4.
// Rows 0 and 8 still share bank (4-way conflict), but that's 4x better than 16x.
constexpr int SMEM_ROW_PAD_BF16 = 8;
constexpr int STRIDE_QK_BF16    = D_QK + SMEM_ROW_PAD_BF16;    // 584

struct SmemLayout {
  static constexpr int Q_BYTES = B_H    * STRIDE_QK_BF16 * 2;
  static constexpr int K_BYTES = B_TOPK * STRIDE_QK_BF16 * 2;
  static constexpr int S_BYTES = B_H    * B_TOPK * 4;
  static constexpr int P_BYTES = B_H    * B_TOPK * 2;
  static constexpr int TOTAL   = Q_BYTES + K_BYTES + S_BYTES + P_BYTES;
  __device__ static bf16*  q(char* p) { return reinterpret_cast<bf16*> (p); }
  __device__ static bf16*  k(char* p) { return reinterpret_cast<bf16*> (p + Q_BYTES); }
  __device__ static float* s(char* p) { return reinterpret_cast<float*>(p + Q_BYTES + K_BYTES); }
  __device__ static bf16*  P(char* p) { return reinterpret_cast<bf16*> (p + Q_BYTES + K_BYTES + S_BYTES); }
};

__device__ __forceinline__ void load_q(bf16* sQ, const bf16* q_nope, const bf16* q_pe, int tid) {
  constexpr int VEC = 8;
  constexpr int THR_PER_ROW = NUM_THREADS / B_H;              // 8
  constexpr int VECS_PER_ROW = D_QK / VEC;                     // 72
  constexpr int VECS_PER_THR = VECS_PER_ROW / THR_PER_ROW;     // 9
  const int row = tid / THR_PER_ROW;
  const int colg = tid % THR_PER_ROW;
  #pragma unroll
  for (int v = 0; v < VECS_PER_THR; ++v) {
    int vidx = colg * VECS_PER_THR + v;
    bf16* dst = sQ + row * STRIDE_QK_BF16 + vidx * VEC;
    const bf16* src = (vidx < D_CKV / VEC) ? (q_nope + row * D_CKV + vidx * VEC)
                                            : (q_pe + row * D_KPE + (vidx - D_CKV / VEC) * VEC);
    cp_async_16_pred(dst, src, true);
  }
}

__device__ __forceinline__ void load_k_block(
    bf16* sK, const int32_t* idx, const bf16* ckv, const bf16* kpe,
    int32_t num_kv, int tid) {
  constexpr int VEC = 8;
  constexpr int THR_PER_ROW = NUM_THREADS / B_TOPK;            // 2
  constexpr int VECS_PER_ROW = D_QK / VEC;                      // 72
  constexpr int VECS_PER_THR = VECS_PER_ROW / THR_PER_ROW;      // 36
  const int row = tid / THR_PER_ROW;
  const int colg = tid % THR_PER_ROW;
  const int32_t tok = idx[row];
  const bool valid = (tok >= 0) && (tok < num_kv);
  #pragma unroll
  for (int v = 0; v < VECS_PER_THR; ++v) {
    int vidx = colg * VECS_PER_THR + v;
    bf16* dst = sK + row * STRIDE_QK_BF16 + vidx * VEC;
    const bf16* src = (vidx < D_CKV / VEC) ? (ckv + tok * D_CKV + vidx * VEC)
                                            : (kpe + tok * D_KPE + (vidx - D_CKV / VEC) * VEC);
    cp_async_16_pred(dst, src, valid);
  }
}

// ---------------- split-kv main kernel ----------------
//
// Processes KV blocks [block_start, block_end) of query `t`.  Produces:
//   o_accum   [num_splits, num_tokens, NUM_HEADS, D_V]  (float)
//   lse_accum [num_splits, num_tokens, NUM_HEADS]       (float, log2 base)
//
// If num_splits == 1 we write straight to the final output instead of accum.

__launch_bounds__(NUM_THREADS, 2)
__global__ void dsa_mla_decode_split_kernel(
    const bf16* q_nope,       // [T, 16, 512]
    const bf16* q_pe,         // [T, 16, 64]
    const bf16* ckv,          // [P, 64, 512]
    const bf16* kpe,          // [P, 64, 64]
    const int32_t* indices,   // [T, 2048]
    float* o_accum,           // [S, T, 16, 512]  (or null if num_splits==1 → write out)
    float* lse_accum,         // [S, T, 16]        (or null if num_splits==1 → write lse)
    bf16*  final_out,         // [T, 16, 512]      (used if num_splits==1)
    float* final_lse,         // [T, 16]           (used if num_splits==1)
    int32_t num_tokens,
    int32_t num_kv,
    int32_t num_splits,
    float sm_scale_log2)
{
  const int bid     = blockIdx.x;
  const int t       = bid / num_splits;
  const int split   = bid % num_splits;
  if (t >= num_tokens) return;

  // Compute [block_start, block_end) for this split.
  // Split blocks roughly evenly: base = NUM_KV_BLOCKS / num_splits, extras = NUM_KV_BLOCKS % num_splits.
  const int base    = NUM_KV_BLOCKS / num_splits;
  const int extras  = NUM_KV_BLOCKS % num_splits;
  const int blk_start = split * base + (split < extras ? split : extras);
  const int blk_end   = blk_start + base + (split < extras ? 1 : 0);
  if (blk_start >= blk_end) return;

  const int tid     = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane    = tid % 32;

  extern __shared__ __align__(16) char smem[];
  bf16*  sQ = SmemLayout::q(smem);
  bf16*  sK = SmemLayout::k(smem);
  float* sS = SmemLayout::s(smem);
  bf16*  sP = SmemLayout::P(smem);

  __shared__ float rowmax_s[B_H];
  __shared__ float rowsum_s[B_H];
  if (tid < B_H) { rowmax_s[tid] = -INFINITY; rowsum_s[tid] = 0.f; }

  // Per-thread O register block: 16 N-sub-tiles × 4 fp32 (mma D layout) = 64 floats.
  // Warp w owns DV_PER_WARP=128 D_V cols across its 16 N-sub-tiles. Per thread,
  // each rO[n_sub] holds a 2x2 fragment: rows {lane/4, lane/4+8}, cols {2*(lane%4), +1}.
  float rO[16][4];
  #pragma unroll
  for (int n_sub = 0; n_sub < 16; ++n_sub) {
    #pragma unroll
    for (int c = 0; c < 4; ++c) rO[n_sub][c] = 0.f;
  }

  const bf16*  q_nope_t = q_nope + t * NUM_HEADS * D_CKV;
  const bf16*  q_pe_t   = q_pe   + t * NUM_HEADS * D_KPE;
  const int32_t* idx_t  = indices + t * TOPK;

  load_q(sQ, q_nope_t, q_pe_t, tid);
  cp_async_commit();

  for (int b = blk_start; b < blk_end; ++b) {
    const int32_t* idx_block = idx_t + b * B_TOPK;
    load_k_block(sK, idx_block, ckv, kpe, num_kv, tid);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    bf16* sKb = sK;

    // QK^T via tensor cores. Each warp computes a M=16 × N=16 slice of S, with
    // N partitioned across the 4 warps. Per warp we split N=16 into 2 mma N-tiles
    // (each N=8) and walk K=576 in 36 K-tiles of K=16, accumulating into a per-
    // thread D[2 N-sub][4 fp32] register block.
    //
    //   A = sQ tile [16, K=16]  — row-major, mma.A "row" layout fits directly.
    //   B = K^T tile [K=16, N=8] — sK[n][k] is naturally K-contiguous-per-N → fits
    //                              mma.B "col" layout with no transpose needed.
    {
      const int row_t  = lane / 4;          // 0..7
      const int col_t  = (lane % 4) * 4;    // 0,4,8,12 (K-offset within tile)
      const int dcol_t = (lane % 4) * 2;    // D-col offset within an N-sub-tile
      const int warp_n_base = warp_id * 16;

      float D[2][4];
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        D[i][0] = D[i][1] = D[i][2] = D[i][3] = 0.f;
      }

      for (int k_base = 0; k_base < D_QK; k_base += 16) {
        // A: rows {row_t, row_t+8} of sQ at cols [k_base+col_t .. +col_t+3].
        uint32_t a0 = *reinterpret_cast<const uint32_t*>(
            &sQ[row_t       * STRIDE_QK_BF16 + k_base + col_t + 0]);
        uint32_t a1 = *reinterpret_cast<const uint32_t*>(
            &sQ[row_t       * STRIDE_QK_BF16 + k_base + col_t + 2]);
        uint32_t a2 = *reinterpret_cast<const uint32_t*>(
            &sQ[(row_t + 8) * STRIDE_QK_BF16 + k_base + col_t + 0]);
        uint32_t a3 = *reinterpret_cast<const uint32_t*>(
            &sQ[(row_t + 8) * STRIDE_QK_BF16 + k_base + col_t + 2]);

        #pragma unroll
        for (int n_sub = 0; n_sub < 2; ++n_sub) {
          const int n_local = warp_n_base + n_sub * 8 + row_t;  // mma.B col index
          uint32_t b0 = *reinterpret_cast<const uint32_t*>(
              &sKb[n_local * STRIDE_QK_BF16 + k_base + col_t + 0]);
          uint32_t b1 = *reinterpret_cast<const uint32_t*>(
              &sKb[n_local * STRIDE_QK_BF16 + k_base + col_t + 2]);
          mma_m16n8k16_bf16_f32(D[n_sub][0], D[n_sub][1], D[n_sub][2], D[n_sub][3],
                                a0, a1, a2, a3, b0, b1);
        }
      }

      // Spill D fragments to sS (which the softmax warp will read row-major).
      #pragma unroll
      for (int n_sub = 0; n_sub < 2; ++n_sub) {
        const int col_base_d = warp_n_base + n_sub * 8 + dcol_t;
        sS[ row_t      * B_TOPK + col_base_d + 0] = D[n_sub][0];
        sS[ row_t      * B_TOPK + col_base_d + 1] = D[n_sub][1];
        sS[(row_t + 8) * B_TOPK + col_base_d + 0] = D[n_sub][2];
        sS[(row_t + 8) * B_TOPK + col_base_d + 1] = D[n_sub][3];
      }
    }
    __syncthreads();

    // Online softmax + sP (single warp does all 16 rows; fast enough).
    __shared__ float scale_o_bcast[B_H];
    if (warp_id == 0 && lane < B_H) {
      int r = lane;
      float new_max = rowmax_s[r];
      #pragma unroll
      for (int c = 0; c < B_TOPK; ++c) {
        float v = sS[r * B_TOPK + c] * sm_scale_log2;
        int32_t tok = idx_block[c];
        if (!(tok >= 0 && tok < num_kv)) v = -INFINITY;
        sS[r * B_TOPK + c] = v;
        new_max = fmaxf(new_max, v);
      }
      float old_max = rowmax_s[r];
      float scale_old = (old_max == -INFINITY) ? 1.f : exp2f(old_max - new_max);
      float sum = 0.f;
      #pragma unroll
      for (int c = 0; c < B_TOPK; ++c) {
        float ex = (sS[r * B_TOPK + c] == -INFINITY) ? 0.f : exp2f(sS[r * B_TOPK + c] - new_max);
        sum += ex;
        sP[r * B_TOPK + c] = __float2bfloat16_rn(ex);
      }
      rowsum_s[r] = rowsum_s[r] * scale_old + sum;
      rowmax_s[r] = new_max;
      scale_o_bcast[r] = scale_old;
    }
    __syncthreads();

    // Rescale rO. With tensor-core layout, rO[n_sub][0..1] correspond to row =
    // (lane/4) and rO[n_sub][2..3] to row = (lane/4)+8 — so a single broadcast
    // pair per thread covers all 16 N-sub-tiles.
    {
      const int row_t = lane / 4;
      const float so0 = scale_o_bcast[row_t];
      const float so1 = scale_o_bcast[row_t + 8];
      #pragma unroll
      for (int n_sub = 0; n_sub < 16; ++n_sub) {
        rO[n_sub][0] *= so0;
        rO[n_sub][1] *= so0;
        rO[n_sub][2] *= so1;
        rO[n_sub][3] *= so1;
      }
    }

    // PV via tensor cores. Each warp owns N = [warp_id*128, +128) of the output.
    // We split N=128 into 16 mma N-sub-tiles (each N=8) and walk K=64 in 4 K-tiles.
    //
    //   A = sP tile [16, K=16] — row-major, mma.A "row" fits directly.
    //   B = V tile [K=16, N=8] — V is sKb[k][n] (K-major in our smem). mma.B
    //     wants col-layout (N-outer in memory), so each thread reads 4 strided
    //     bf16 values along K at one fixed N column. Not as efficient as
    //     ldmatrix.trans but stays within plain PTX and avoids extra smem.
    {
      const int row_t  = lane / 4;
      const int col_t  = (lane % 4) * 4;
      const int warp_n_base = warp_id * DV_PER_WARP;  // 128 cols per warp

      #pragma unroll 1
      for (int k_base = 0; k_base < B_TOPK; k_base += 16) {
        uint32_t a0 = *reinterpret_cast<const uint32_t*>(
            &sP[row_t       * B_TOPK + k_base + col_t + 0]);
        uint32_t a1 = *reinterpret_cast<const uint32_t*>(
            &sP[row_t       * B_TOPK + k_base + col_t + 2]);
        uint32_t a2 = *reinterpret_cast<const uint32_t*>(
            &sP[(row_t + 8) * B_TOPK + k_base + col_t + 0]);
        uint32_t a3 = *reinterpret_cast<const uint32_t*>(
            &sP[(row_t + 8) * B_TOPK + k_base + col_t + 2]);

        #pragma unroll
        for (int n_sub = 0; n_sub < 16; ++n_sub) {
          const int n_local = warp_n_base + n_sub * 8 + row_t;
          // Strided B load: 4 bf16 from 4 K-rows at fixed N column. Pack into
          // 2 .b32 directly via raw bit views to avoid type-pun aliasing.
          const uint16_t* sKb_u16 = reinterpret_cast<const uint16_t*>(sKb);
          uint16_t v0 = sKb_u16[(k_base + col_t + 0) * STRIDE_QK_BF16 + n_local];
          uint16_t v1 = sKb_u16[(k_base + col_t + 1) * STRIDE_QK_BF16 + n_local];
          uint16_t v2 = sKb_u16[(k_base + col_t + 2) * STRIDE_QK_BF16 + n_local];
          uint16_t v3 = sKb_u16[(k_base + col_t + 3) * STRIDE_QK_BF16 + n_local];
          uint32_t b0 = (uint32_t)v0 | ((uint32_t)v1 << 16);
          uint32_t b1 = (uint32_t)v2 | ((uint32_t)v3 << 16);
          mma_m16n8k16_bf16_f32(rO[n_sub][0], rO[n_sub][1], rO[n_sub][2], rO[n_sub][3],
                                a0, a1, a2, a3, b0, b1);
        }
      }
    }
    __syncthreads();
  }

  // ---- Epilogue ----
  // rO is now in mma's D layout: rO[n_sub][0..1] cover (row=row_t, col=col_d, col_d+1)
  // and rO[n_sub][2..3] cover (row=row_t+8, col_d, col_d+1) within an N-sub-tile;
  // each warp owns DV_PER_WARP=128 cols across 16 N-sub-tiles.
  {
    const int row_t  = lane / 4;
    const int dcol_t = (lane % 4) * 2;
    const int warp_n_base = warp_id * DV_PER_WARP;
    const float rs0 = rowsum_s[row_t];
    const float rs1 = rowsum_s[row_t + 8];
    const float inv0 = (rs0 == 0.f) ? 0.f : (1.f / rs0);
    const float inv1 = (rs1 == 0.f) ? 0.f : (1.f / rs1);

    if (num_splits == 1) {
      bf16* out_t = final_out + t * NUM_HEADS * D_V;
      #pragma unroll
      for (int n_sub = 0; n_sub < 16; ++n_sub) {
        const int col_global = warp_n_base + n_sub * 8 + dcol_t;
        bf162 lo = __floats2bfloat162_rn(rO[n_sub][0] * inv0, rO[n_sub][1] * inv0);
        bf162 hi = __floats2bfloat162_rn(rO[n_sub][2] * inv1, rO[n_sub][3] * inv1);
        *reinterpret_cast<bf162*>(&out_t[ row_t      * D_V + col_global]) = lo;
        *reinterpret_cast<bf162*>(&out_t[(row_t + 8) * D_V + col_global]) = hi;
      }
      if (final_lse != nullptr && warp_id == 0 && lane < B_H) {
        float s = rowsum_s[lane];
        float m = rowmax_s[lane];
        float v = (m == -INFINITY || s == 0.f) ? -INFINITY : (log2f(s) + m);
        final_lse[t * NUM_HEADS + lane] = v;
      }
    } else {
      // Write *normalized* o_accum (rO / rowsum) and lse_accum = log2(rowsum) + rowmax.
      // Combine kernel weights each split by 2^{lse_s - global_max}, sums normalized O, divides.
      float* o_t = o_accum + ((int64_t)split * num_tokens + t) * NUM_HEADS * D_V;
      float* l_t = lse_accum + ((int64_t)split * num_tokens + t) * NUM_HEADS;
      #pragma unroll
      for (int n_sub = 0; n_sub < 16; ++n_sub) {
        const int col_global = warp_n_base + n_sub * 8 + dcol_t;
        o_t[ row_t      * D_V + col_global + 0] = rO[n_sub][0] * inv0;
        o_t[ row_t      * D_V + col_global + 1] = rO[n_sub][1] * inv0;
        o_t[(row_t + 8) * D_V + col_global + 0] = rO[n_sub][2] * inv1;
        o_t[(row_t + 8) * D_V + col_global + 1] = rO[n_sub][3] * inv1;
      }
      if (warp_id == 0 && lane < B_H) {
        float s = rowsum_s[lane];
        float m = rowmax_s[lane];
        l_t[lane] = (m == -INFINITY || s == 0.f) ? -INFINITY : (log2f(s) + m);
      }
    }
  }
}

// ---------------- combine kernel ----------------
// Merges per-split (o_accum, lse_accum) into final (output, lse) via log-sum-exp.
// One CTA per (token, head); 128 threads × 4 cols of D_V. Templated on
// NUM_SPLITS (∈ {2,4,8,16,32}) so all NUM_SPLITS o_accum loads are issued back-
// to-back with no carried dependency, and softmax/accumulate fully unroll.

template <int NUM_SPLITS>
__launch_bounds__(D_V / 4, 4) __global__ void dsa_mla_combine_kernel(
    const float* __restrict__ o_accum,    // [S, T, H, D_V]
    const float* __restrict__ lse_accum,  // [S, T, H]
    bf16*        __restrict__ output,     // [T, H, D_V]
    float*       __restrict__ lse,        // [T, H] or nullptr
    int32_t num_tokens) {
  static_assert(NUM_SPLITS >= 2 && NUM_SPLITS <= NUM_KV_BLOCKS);
  const int th = blockIdx.x;
  const int t = th / NUM_HEADS;
  const int h = th % NUM_HEADS;
  if (t >= num_tokens) return;
  const int tid = threadIdx.x;
  const int col_base = tid * 4;

  __shared__ float s_lse[NUM_SPLITS];
  if (tid < NUM_SPLITS) {
    s_lse[tid] = lse_accum[(int64_t)tid * num_tokens * NUM_HEADS + t * NUM_HEADS + h];
  }

  float o_local[NUM_SPLITS][4];
#pragma unroll
  for (int s = 0; s < NUM_SPLITS; ++s) {
    const float* o_p =
        o_accum + ((int64_t)s * num_tokens + t) * NUM_HEADS * D_V + h * D_V + col_base;
    float4 a = *reinterpret_cast<const float4*>(o_p);
    o_local[s][0] = a.x;
    o_local[s][1] = a.y;
    o_local[s][2] = a.z;
    o_local[s][3] = a.w;
  }

  __syncthreads();

  float local_lse[NUM_SPLITS];
#pragma unroll
  for (int s = 0; s < NUM_SPLITS; ++s) local_lse[s] = s_lse[s];

  float gmax = -INFINITY;
#pragma unroll
  for (int s = 0; s < NUM_SPLITS; ++s) gmax = fmaxf(gmax, local_lse[s]);

  float gsum = 0.f;
  float scales[NUM_SPLITS];
#pragma unroll
  for (int s = 0; s < NUM_SPLITS; ++s) {
    float w = (local_lse[s] == -INFINITY) ? 0.f : exp2f(local_lse[s] - gmax);
    scales[s] = w;
    gsum += w;
  }
  const float inv_gsum = (gsum == 0.f) ? 0.f : __frcp_rn(gsum);

  float4 acc = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int s = 0; s < NUM_SPLITS; ++s) {
    const float w = scales[s] * inv_gsum;
    acc.x += w * o_local[s][0];
    acc.y += w * o_local[s][1];
    acc.z += w * o_local[s][2];
    acc.w += w * o_local[s][3];
  }

  bf16* out_p = output + t * NUM_HEADS * D_V + h * D_V + col_base;
  bf162 lo = __floats2bfloat162_rn(acc.x, acc.y);
  bf162 hi = __floats2bfloat162_rn(acc.z, acc.w);
  *reinterpret_cast<bf162*>(out_p) = lo;
  *reinterpret_cast<bf162*>(out_p + 2) = hi;
  if (lse != nullptr && tid == 0) {
    float v = (gmax == -INFINITY || gsum == 0.f) ? -INFINITY : (log2f(gsum) + gmax);
    lse[t * NUM_HEADS + h] = v;
  }
}

// Choose num_splits — prefer divisors of NUM_KV_BLOCKS for balanced work.
//
// We aim for `T * num_splits` to be ≤ 2 * num_sms so each SM holds at most 2 CTAs
// (the kernel is launch_bounded(2)). At the upper end the overlap of two CTAs on
// one SM hides cp.async / smem latency better than one large CTA per SM.
//
// At low T we cap num_splits at NUM_KV_BLOCKS so each CTA owns ≥1 block.
__host__ inline int choose_num_splits(int num_tokens, int num_sms) {
  // Target 2 CTAs/SM utilization; pick the largest divisor that doesn't blow past it.
  int want = (2 * num_sms + num_tokens - 1) / num_tokens;
  int candidates[] = {32, 16, 8, 4, 2, 1};
  for (int c : candidates) {
    if (c <= want) return c;
  }
  return 1;
}

}  // namespace dsa_mla_ns

// ---------------- host entry ----------------
void dsa_mla_decode(
    tvm::ffi::TensorView q_nope,
    tvm::ffi::TensorView q_pe,
    tvm::ffi::TensorView ckv_cache,
    tvm::ffi::TensorView kpe_cache,
    tvm::ffi::TensorView sparse_indices,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView lse,
    tvm::ffi::TensorView workspace,
    double sm_scale) {
  using namespace dsa_mla_ns;
  using namespace host;

  SymbolicSize Ntok{"num_tokens"};
  SymbolicSize Npage{"num_pages"};
  SymbolicDevice dev_;

  TensorMatcher({Ntok, (int64_t)NUM_HEADS, (int64_t)D_CKV}).with_dtype<bf16_t>().with_device<kDLCUDA>(dev_).verify(q_nope);
  TensorMatcher({Ntok, (int64_t)NUM_HEADS, (int64_t)D_KPE}).with_dtype<bf16_t>().with_device<kDLCUDA>(dev_).verify(q_pe);
  TensorMatcher({Npage, (int64_t)PAGE_SIZE, (int64_t)D_CKV}).with_dtype<bf16_t>().with_device<kDLCUDA>(dev_).verify(ckv_cache);
  TensorMatcher({Npage, (int64_t)PAGE_SIZE, (int64_t)D_KPE}).with_dtype<bf16_t>().with_device<kDLCUDA>(dev_).verify(kpe_cache);
  TensorMatcher({Ntok, (int64_t)TOPK}).with_dtype<int32_t>().with_device<kDLCUDA>(dev_).verify(sparse_indices);
  TensorMatcher({Ntok, (int64_t)NUM_HEADS, (int64_t)D_V}).with_dtype<bf16_t>().with_device<kDLCUDA>(dev_).verify(output);
  // lse is optional: a 1-D zero-numel tensor signals "skip writing lse".
  const bool want_lse = (lse.numel() > 0);
  if (want_lse) {
    TensorMatcher({Ntok, (int64_t)NUM_HEADS}).with_dtype<float>().with_device<kDLCUDA>(dev_).verify(lse);
  }
  // workspace is bytes (uint8) 1D, validated loosely.
  RuntimeCheck(workspace.device().device_type == kDLCUDA, "workspace must be CUDA");

  const int32_t num_tokens = static_cast<int32_t>(Ntok.unwrap());
  const int32_t num_pages  = static_cast<int32_t>(Npage.unwrap());
  const int32_t num_kv_tokens = num_pages * PAGE_SIZE;
  const DLDevice device = dev_.unwrap();
  const cudaStream_t stream = LaunchKernel::resolve_device(device);
  if (num_tokens == 0) return;

  int sm_count = 0;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device.device_id);
  if (sm_count <= 0) sm_count = 148;

  const int num_splits = choose_num_splits(num_tokens, sm_count);

  constexpr size_t smem_bytes = SmemLayout::TOTAL;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(
        (const void*)dsa_mla_decode_split_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));
    attr_set = true;
  }

  const float sm_scale_log2 = static_cast<float>(sm_scale) * 1.4426950408889634f;

  bf16* out_ptr = static_cast<bf16*>(output.data_ptr());
  float* lse_ptr = want_lse ? static_cast<float*>(lse.data_ptr()) : nullptr;

  if (num_splits == 1) {
    dim3 grid(num_tokens);
    dim3 block(NUM_THREADS);
    dsa_mla_decode_split_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<const bf16*>(q_nope.data_ptr()),
        static_cast<const bf16*>(q_pe.data_ptr()),
        static_cast<const bf16*>(ckv_cache.data_ptr()),
        static_cast<const bf16*>(kpe_cache.data_ptr()),
        static_cast<const int32_t*>(sparse_indices.data_ptr()),
        nullptr, nullptr,
        out_ptr, lse_ptr,
        num_tokens, num_kv_tokens, 1, sm_scale_log2);
  } else {
    // Carve workspace: [o_accum float[S,T,16,512]][lse_accum float[S,T,16]] with alignment.
    size_t o_bytes   = (size_t)num_splits * num_tokens * NUM_HEADS * D_V * sizeof(float);
    size_t lse_bytes = (size_t)num_splits * num_tokens * NUM_HEADS     * sizeof(float);
    size_t total_need = o_bytes + lse_bytes;
    RuntimeCheck(
        (size_t)workspace.size(0) >= total_need,
        "workspace too small: have ", workspace.size(0), " need ", total_need);
    char* ws = static_cast<char*>(workspace.data_ptr());
    float* o_accum   = reinterpret_cast<float*>(ws);
    float* lse_accum = reinterpret_cast<float*>(ws + o_bytes);

    dim3 grid(num_tokens * num_splits);
    dim3 block(NUM_THREADS);
    dsa_mla_decode_split_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<const bf16*>(q_nope.data_ptr()),
        static_cast<const bf16*>(q_pe.data_ptr()),
        static_cast<const bf16*>(ckv_cache.data_ptr()),
        static_cast<const bf16*>(kpe_cache.data_ptr()),
        static_cast<const int32_t*>(sparse_indices.data_ptr()),
        o_accum, lse_accum,
        nullptr, nullptr,
        num_tokens, num_kv_tokens, num_splits, sm_scale_log2);

    dim3 cgrid(num_tokens * NUM_HEADS);
    dim3 cblk(D_V / 4);   // 128 threads × 4 cols
#define LAUNCH_COMBINE(NS)                                                                \
  dsa_mla_combine_kernel<NS><<<cgrid, cblk, 0, stream>>>(                                 \
      o_accum, lse_accum, out_ptr, lse_ptr, num_tokens);
    switch (num_splits) {
      case 2:  LAUNCH_COMBINE(2);  break;
      case 4:  LAUNCH_COMBINE(4);  break;
      case 8:  LAUNCH_COMBINE(8);  break;
      case 16: LAUNCH_COMBINE(16); break;
      case 32: LAUNCH_COMBINE(32); break;
      default:
        RuntimeCheck(false, "dsa_mla: unsupported num_splits ", num_splits);
    }
#undef LAUNCH_COMBINE
  }
}
