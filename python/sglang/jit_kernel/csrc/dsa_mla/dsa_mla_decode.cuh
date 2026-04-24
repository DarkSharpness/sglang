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
__device__ __forceinline__ void cp_async_16_pred(void* smem_dst, const void* gmem_src, bool pred) {
  uint32_t s = smem_u32(smem_dst);
  if (pred) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" : : "r"(s), "l"(gmem_src));
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, 0;\n" : : "r"(s), "l"(gmem_src));
  }
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::); }

struct SmemLayout {
  static constexpr int Q_BYTES = B_H    * D_QK * 2;
  static constexpr int K_BYTES = B_TOPK * D_QK * 2;
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
    bf16* dst = sQ + row * D_QK + vidx * VEC;
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
    bf16* dst = sK + row * D_QK + vidx * VEC;
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

__launch_bounds__(NUM_THREADS, 1)
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

  // Per-thread O register block: 16 rows × 4 cols = 64 floats.
  // Warp w: owns cols [w*DV_PER_WARP + lane*4 .. +4).
  float rO[B_H][4];
  #pragma unroll
  for (int r = 0; r < B_H; ++r) {
    #pragma unroll
    for (int c = 0; c < 4; ++c) rO[r][c] = 0.f;
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

    // QK^T: warp w computes S[:, w*16 : w*16+16].
    {
      const int row = lane % 16;
      const int col_base = (lane / 16) * 8;
      #pragma unroll
      for (int c = 0; c < 8; ++c) {
        int kv_row = warp_id * 16 + col_base + c;
        float acc = 0.f;
        #pragma unroll
        for (int k = 0; k < D_QK; k += 8) {
          bf162 q0 = *reinterpret_cast<bf162*>(&sQ[row * D_QK + k]);
          bf162 q1 = *reinterpret_cast<bf162*>(&sQ[row * D_QK + k + 2]);
          bf162 q2 = *reinterpret_cast<bf162*>(&sQ[row * D_QK + k + 4]);
          bf162 q3 = *reinterpret_cast<bf162*>(&sQ[row * D_QK + k + 6]);
          bf162 k0 = *reinterpret_cast<bf162*>(&sK[kv_row * D_QK + k]);
          bf162 k1 = *reinterpret_cast<bf162*>(&sK[kv_row * D_QK + k + 2]);
          bf162 k2 = *reinterpret_cast<bf162*>(&sK[kv_row * D_QK + k + 4]);
          bf162 k3 = *reinterpret_cast<bf162*>(&sK[kv_row * D_QK + k + 6]);
          float2 q0f = __bfloat1622float2(q0);
          float2 q1f = __bfloat1622float2(q1);
          float2 q2f = __bfloat1622float2(q2);
          float2 q3f = __bfloat1622float2(q3);
          float2 k0f = __bfloat1622float2(k0);
          float2 k1f = __bfloat1622float2(k1);
          float2 k2f = __bfloat1622float2(k2);
          float2 k3f = __bfloat1622float2(k3);
          acc += q0f.x * k0f.x + q0f.y * k0f.y
               + q1f.x * k1f.x + q1f.y * k1f.y
               + q2f.x * k2f.x + q2f.y * k2f.y
               + q3f.x * k3f.x + q3f.y * k3f.y;
        }
        sS[row * B_TOPK + warp_id * 16 + col_base + c] = acc;
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

    // Rescale rO
    #pragma unroll
    for (int r = 0; r < B_H; ++r) {
      float so = scale_o_bcast[r];
      #pragma unroll
      for (int c = 0; c < 4; ++c) rO[r][c] *= so;
    }

    // PV: O[row, col] += sum_k P[row, k] * V[k, col]
    {
      const int col_base_warp = warp_id * DV_PER_WARP + lane * 4;
      #pragma unroll
      for (int r = 0; r < B_H; ++r) {
        float acc0 = rO[r][0], acc1 = rO[r][1], acc2 = rO[r][2], acc3 = rO[r][3];
        #pragma unroll
        for (int k = 0; k < B_TOPK; ++k) {
          float p = __bfloat162float(sP[r * B_TOPK + k]);
          bf162 v01 = *reinterpret_cast<bf162*>(&sK[k * D_QK + col_base_warp]);
          bf162 v23 = *reinterpret_cast<bf162*>(&sK[k * D_QK + col_base_warp + 2]);
          float2 v01f = __bfloat1622float2(v01);
          float2 v23f = __bfloat1622float2(v23);
          acc0 += p * v01f.x;
          acc1 += p * v01f.y;
          acc2 += p * v23f.x;
          acc3 += p * v23f.y;
        }
        rO[r][0] = acc0; rO[r][1] = acc1; rO[r][2] = acc2; rO[r][3] = acc3;
      }
    }
    __syncthreads();
  }

  // ---- Epilogue ----
  if (num_splits == 1) {
    bf16* out_t = final_out + t * NUM_HEADS * D_V;
    const int col_base_warp = warp_id * DV_PER_WARP + lane * 4;
    #pragma unroll
    for (int r = 0; r < B_H; ++r) {
      float inv = (rowsum_s[r] == 0.f) ? 0.f : 1.f / rowsum_s[r];
      bf162 lo = __floats2bfloat162_rn(rO[r][0] * inv, rO[r][1] * inv);
      bf162 hi = __floats2bfloat162_rn(rO[r][2] * inv, rO[r][3] * inv);
      *reinterpret_cast<bf162*>(&out_t[r * D_V + col_base_warp])     = lo;
      *reinterpret_cast<bf162*>(&out_t[r * D_V + col_base_warp + 2]) = hi;
    }
    if (warp_id == 0 && lane < B_H) {
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
    const int col_base_warp = warp_id * DV_PER_WARP + lane * 4;
    #pragma unroll
    for (int r = 0; r < B_H; ++r) {
      const float rs = rowsum_s[r];
      const float inv = (rs == 0.f) ? 0.f : (1.f / rs);
      o_t[r * D_V + col_base_warp    ] = rO[r][0] * inv;
      o_t[r * D_V + col_base_warp + 1] = rO[r][1] * inv;
      o_t[r * D_V + col_base_warp + 2] = rO[r][2] * inv;
      o_t[r * D_V + col_base_warp + 3] = rO[r][3] * inv;
    }
    if (warp_id == 0 && lane < B_H) {
      float s = rowsum_s[lane];
      float m = rowmax_s[lane];
      l_t[lane] = (m == -INFINITY || s == 0.f) ? -INFINITY : (log2f(s) + m);
    }
  }
}

// ---------------- combine kernel ----------------
// Merges per-split (o_accum, lse_accum) into final (output, lse) via log-sum-exp reduction.
//
// Each CTA handles one (token, head) pair.  NUM_HEADS=16 tokens=Ntok → Ntok*16 blocks.
//
// For a single head:
//   lse_final = log2(sum_s 2^{lse_accum[s] - max_lse})  +  max_lse
//   out_final = sum_s o_accum[s] * 2^{lse_accum[s] - max_lse_noscale}   (where noscale = used row-max)
//   actually: o_accum[s] is already un-normalized (raw softmaxed * V, not divided by rowsum).
//   rowsum_for_split_s = exp2(lse_accum[s] - (log2(rowsum_s) + max_s))
//   Hmm — simpler to think in full terms:
//     lse_accum[s] = log2(sum_j exp2(logit - max))+ max  → equivalent to log2(unnormalized_denom) + max
//       where unnormalized_denom = sum exp(logit * ln2) = sum exp2(logit - max) * 2^max ≈ sum*2^max
//
//   We have per-split:
//     rowsum_s[split]  = sum_j exp2(logit_scaled_j - local_max_s)
//     rowmax_s[split]  = local_max_s
//     o_accum[split]   = sum_j exp2(logit_j - local_max_s) * V[j]    (unnormalized by rowsum)
//
//   And we saved lse_s = log2(rowsum_s) + local_max_s.
//
//   Combine across splits:
//     global_max = max_s(local_max_s)
//     w_s = 2^{local_max_s - global_max}
//     global_sum = sum_s (rowsum_s * w_s)  =  sum_s 2^{lse_s - global_max}
//     o_final_unnorm = sum_s (o_accum[s] * w_s)
//     o_final = o_final_unnorm / global_sum
//     lse_final = log2(global_sum) + global_max
//
// Since lse_s = log2(rowsum_s) + local_max_s = log2(rowsum_s * 2^local_max_s),
// 2^{lse_s - global_max} = rowsum_s * 2^{local_max_s - global_max} = rowsum_s * w_s.
//
// And o_accum[s] is unnormalized (before dividing by rowsum_s), so
// (o_accum[s] * w_s) has the same weighting as rowsum_s * w_s.  Good.

__global__ void dsa_mla_combine_kernel(
    const float* __restrict__ o_accum,    // [S, T, H, D_V]
    const float* __restrict__ lse_accum,  // [S, T, H]
    bf16*       __restrict__ output,       // [T, H, D_V]
    float*      __restrict__ lse,          // [T, H]
    int32_t num_splits,
    int32_t num_tokens)
{
  const int th = blockIdx.x;
  const int t = th / NUM_HEADS;
  const int h = th % NUM_HEADS;
  if (t >= num_tokens) return;

  const int tid = threadIdx.x;
  constexpr int CTA = 128;  // 1 CTA of 128 threads, each handles D_V/128 = 4 cols.

  // Step 1: global_max over splits.
  float gmax = -INFINITY;
  for (int s = 0; s < num_splits; ++s) {
    float lv = lse_accum[(int64_t)s * num_tokens * NUM_HEADS + t * NUM_HEADS + h];
    gmax = fmaxf(gmax, lv);
  }

  // Step 2: per-split w_s = 2^{lse_s - gmax}, global_sum.
  float gsum = 0.f;
  for (int s = 0; s < num_splits; ++s) {
    float lv = lse_accum[(int64_t)s * num_tokens * NUM_HEADS + t * NUM_HEADS + h];
    if (lv != -INFINITY) {
      gsum += exp2f(lv - gmax);
    }
  }

  // Step 3: per col_group, accumulate o.
  // Each thread handles D_V/CTA = 4 consecutive cols.
  const int col_base = tid * 4;
  float4 acc = {0.f, 0.f, 0.f, 0.f};
  for (int s = 0; s < num_splits; ++s) {
    float lv = lse_accum[(int64_t)s * num_tokens * NUM_HEADS + t * NUM_HEADS + h];
    if (lv == -INFINITY) continue;
    // w = 2^{lv - gmax}  — but lv = log2(rowsum_s) + local_max_s, so
    // w = rowsum_s * 2^{local_max_s - gmax}.  o_accum has the *unnormalized* running sum
    // sum_j exp2(logit - local_max_s) * V = (rowsum_s-weighted).
    // Net: o_final unnormalized contribution = o_accum * 2^{local_max_s - gmax}.
    // We only stored lv = log2(rowsum_s) + local_max_s, not local_max_s alone.  But we can
    // recover exp2(local_max_s - gmax) = exp2(lv - gmax) / rowsum_s.  Hmm, we don't have rowsum.
    //
    // Alternative: don't store the raw unnormalized o_accum.  Store normalized o_accum/rowsum
    // instead.  Then o_final = sum_s (o_accum_norm * 2^{lv - gmax}) / sum_s 2^{lv - gmax}.
    //
    // That's much simpler if the split kernel NORMALIZES o_accum before writing.
    // So I'll change the split kernel to write o_accum = rO / rowsum_s.
    const float w = exp2f(lv - gmax);
    const float* o_t = o_accum + ((int64_t)s * num_tokens + t) * NUM_HEADS * D_V + h * D_V;
    float4 a = *reinterpret_cast<const float4*>(&o_t[col_base]);
    acc.x += a.x * w;
    acc.y += a.y * w;
    acc.z += a.z * w;
    acc.w += a.w * w;
  }
  const float inv_gsum = (gsum == 0.f) ? 0.f : (1.f / gsum);
  acc.x *= inv_gsum;
  acc.y *= inv_gsum;
  acc.z *= inv_gsum;
  acc.w *= inv_gsum;

  bf16* out_p = output + t * NUM_HEADS * D_V + h * D_V + col_base;
  bf162 lo = __floats2bfloat162_rn(acc.x, acc.y);
  bf162 hi = __floats2bfloat162_rn(acc.z, acc.w);
  *reinterpret_cast<bf162*>(out_p    ) = lo;
  *reinterpret_cast<bf162*>(out_p + 2) = hi;

  if (tid == 0) {
    float v = (gmax == -INFINITY || gsum == 0.f) ? -INFINITY : (log2f(gsum) + gmax);
    lse[t * NUM_HEADS + h] = v;
  }
}

// Choose num_splits — return a divisor of NUM_KV_BLOCKS (=32) bounded by SM count.
__host__ inline int choose_num_splits(int num_tokens, int num_sms) {
  int want = (num_sms + num_tokens - 1) / num_tokens;   // CTAs per token
  // Round down to the nearest divisor of NUM_KV_BLOCKS in {32,16,8,4,2,1}.
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
  TensorMatcher({Ntok, (int64_t)NUM_HEADS}).with_dtype<float>().with_device<kDLCUDA>(dev_).verify(lse);

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
        static_cast<bf16*>(output.data_ptr()),
        static_cast<float*>(lse.data_ptr()),
        num_tokens, num_kv_tokens, 1, sm_scale_log2);
  } else {
    // Allocate scratch via thrust/torch allocator is not readily available here; use per-call
    // cudaMalloc (will be cached by the allocator).  For correctness we go with plain malloc on
    // the device side first; real path should reuse a workspace.
    // NB: this path is NOT zero-overhead; we'll replace with a workspace in a follow-up.
    size_t o_bytes   = (size_t)num_splits * num_tokens * NUM_HEADS * D_V * sizeof(float);
    size_t lse_bytes = (size_t)num_splits * num_tokens * NUM_HEADS     * sizeof(float);
    float* o_accum = nullptr;
    float* lse_accum = nullptr;
    cudaMallocAsync((void**)&o_accum, o_bytes, stream);
    cudaMallocAsync((void**)&lse_accum, lse_bytes, stream);

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
    dsa_mla_combine_kernel<<<cgrid, cblk, 0, stream>>>(
        o_accum, lse_accum,
        static_cast<bf16*>(output.data_ptr()),
        static_cast<float*>(lse.data_ptr()),
        num_splits, num_tokens);

    cudaFreeAsync(o_accum, stream);
    cudaFreeAsync(lse_accum, stream);
  }
}
