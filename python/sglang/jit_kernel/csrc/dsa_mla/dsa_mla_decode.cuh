// DeepSeek Sparse Attention (DSA) MLA decode kernel — SM100 (B200) / h_q = 16.
//
// V0 (baseline, no mma): naive FMA loop for correctness. Slow but easy to debug.

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

constexpr int NUM_HEADS   = 16;
constexpr int D_CKV       = 512;
constexpr int D_KPE       = 64;
constexpr int D_QK        = D_CKV + D_KPE;
constexpr int D_V         = 512;
constexpr int PAGE_SIZE   = 64;
constexpr int TOPK        = 2048;

constexpr int B_H         = 16;
constexpr int B_TOPK      = 64;
constexpr int NUM_WARPS   = 4;
constexpr int NUM_THREADS = NUM_WARPS * 32;
constexpr int NUM_KV_BLOCKS = TOPK / B_TOPK;

constexpr int DV_PER_WARP = D_V / NUM_WARPS;  // 128

// -------- PTX helpers --------
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
  static constexpr int Q_BYTES = B_H    * D_QK * 2;  // 18,432
  static constexpr int K_BYTES = B_TOPK * D_QK * 2;  // 73,728
  static constexpr int S_BYTES = B_H    * B_TOPK * 4; //  4,096
  static constexpr int P_BYTES = B_H    * B_TOPK * 2; //  2,048
  static constexpr int TOTAL   = Q_BYTES + K_BYTES + S_BYTES + P_BYTES;
  __device__ static bf16*  q(char* p) { return reinterpret_cast<bf16*> (p); }
  __device__ static bf16*  k(char* p) { return reinterpret_cast<bf16*> (p + Q_BYTES); }
  __device__ static float* s(char* p) { return reinterpret_cast<float*>(p + Q_BYTES + K_BYTES); }
  __device__ static bf16*  P(char* p) { return reinterpret_cast<bf16*> (p + Q_BYTES + K_BYTES + S_BYTES); }
};

__device__ __forceinline__ void load_q(bf16* sQ, const bf16* q_nope, const bf16* q_pe, int tid) {
  constexpr int VEC = 8;
  constexpr int ROWS = B_H;
  constexpr int THR_PER_ROW = NUM_THREADS / ROWS;  // 8
  constexpr int VECS_PER_ROW = D_QK / VEC;  // 72
  constexpr int VECS_PER_THR = VECS_PER_ROW / THR_PER_ROW;  // 9
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
  constexpr int ROWS = B_TOPK;
  constexpr int THR_PER_ROW = NUM_THREADS / ROWS;  // 2
  constexpr int VECS_PER_ROW = D_QK / VEC;  // 72
  constexpr int VECS_PER_THR = VECS_PER_ROW / THR_PER_ROW;  // 36
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

__launch_bounds__(NUM_THREADS, 1)
__global__ void dsa_mla_decode_kernel_v0(
    const bf16* q_nope, const bf16* q_pe,
    const bf16* ckv, const bf16* kpe,
    const int32_t* indices,
    bf16* output, float* lse_out,
    int32_t num_tokens, int32_t num_kv, float sm_scale_log2) {
  const int t = blockIdx.x;
  if (t >= num_tokens) return;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane = tid % 32;

  extern __shared__ __align__(16) char smem[];
  bf16*  sQ = SmemLayout::q(smem);
  bf16*  sK = SmemLayout::k(smem);
  float* sS = SmemLayout::s(smem);
  bf16*  sP = SmemLayout::P(smem);

  __shared__ float rowmax_s[B_H];
  __shared__ float rowsum_s[B_H];
  if (tid < B_H) { rowmax_s[tid] = -INFINITY; rowsum_s[tid] = 0.f; }

  // O accumulator (FP32) in registers.  Each warp owns 128 cols of d_v.
  // Layout: each thread handles a contiguous block of rows × cols.  For 16 rows × 128 cols =
  // 2048 elements / 32 threads per warp = 64 elements per thread.
  //
  // Simple partition: thread owns rows = lane/2, and cols = (lane%2)*64..(lane%2)*64+64 within warp.
  // Wait — this would be 1 row per thread × 64 cols = 64 elts ✓.  But some threads get the same row.
  //
  // Cleaner: 32 threads split 16×128 = 2048 elements → 64 per thread.  Each thread owns a 2-row ×
  // 32-col slab? Let's do: thread lane owns rows (lane / 8)   [0..3] and cols (lane % 8) * 16 + [0..15]
  //    → 4 rows × 16 cols = 64 ✓.
  // No wait — 32 threads × (rows × cols) = 16 × 128. 16 × 128 / 32 = 64 ✓.
  // Split: 4 rows × 8 threads (col dim).  rows_per_thread = 4 (lane/8), cols_per_thread = 16 (lane%8 * 16).
  //   So 4 warps × 4 rows = 16 rows ✓.  But each warp should cover all 16 rows (M=16), with N=128.
  //   Actually, cleaner for ALL warps to cover ALL 16 rows (each warp has different N slice):
  //   Each warp owns rows 0..15 × 128 N-cols = 16×128 = 2048 per warp / 32 threads = 64 per thread.
  //
  // Let each thread own 16 rows × 4 cols = 64 elements.  Layout:
  //   lane l owns cols [l*4, l*4+4) in the warp's 128-col slab.  All 16 rows.
  //   Col global index: warp_id * DV_PER_WARP + lane * 4 + c for c in 0..3.
  //
  // Register layout: rO[row][c] where row in 0..15, c in 0..3.

  float rO[B_H][4];
  #pragma unroll
  for (int r = 0; r < B_H; ++r) {
    #pragma unroll
    for (int c = 0; c < 4; ++c) rO[r][c] = 0.f;
  }

  const bf16* q_nope_t = q_nope + t * NUM_HEADS * D_CKV;
  const bf16* q_pe_t   = q_pe   + t * NUM_HEADS * D_KPE;
  const int32_t* idx_t = indices + t * TOPK;
  load_q(sQ, q_nope_t, q_pe_t, tid);
  cp_async_commit();

  for (int b = 0; b < NUM_KV_BLOCKS; ++b) {
    const int32_t* idx_block = idx_t + b * B_TOPK;
    load_k_block(sK, idx_block, ckv, kpe, num_kv, tid);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // -------- QK^T: naive FMA.  Warp w computes S[:, w*16 .. (w+1)*16]. --------
    // 16 rows (heads) × 16 kv-cols / 32 threads = 8 elements per thread.
    // Partition: lane owns (row = lane%16) × (col = lane/16)*8 + c for c in 0..7.  So 8 elts/thread.
    // Then 4 warps × 16 cols = 64 cols total ✓.
    {
      const int row = lane % 16;
      const int col_base = (lane / 16) * 8;  // 0 or 8
      #pragma unroll
      for (int c = 0; c < 8; ++c) {
        int kv_row = warp_id * 16 + col_base + c;
        float acc = 0.f;
        #pragma unroll
        for (int k = 0; k < D_QK; k += 8) {
          // 8 bf16 fmas
          bf162 q0 = *reinterpret_cast<bf162*>(&sQ[row * D_QK + k]);
          bf162 q1 = *reinterpret_cast<bf162*>(&sQ[row * D_QK + k + 2]);
          bf162 q2 = *reinterpret_cast<bf162*>(&sQ[row * D_QK + k + 4]);
          bf162 q3 = *reinterpret_cast<bf162*>(&sQ[row * D_QK + k + 6]);
          bf162 kk0 = *reinterpret_cast<bf162*>(&sK[kv_row * D_QK + k]);
          bf162 kk1 = *reinterpret_cast<bf162*>(&sK[kv_row * D_QK + k + 2]);
          bf162 kk2 = *reinterpret_cast<bf162*>(&sK[kv_row * D_QK + k + 4]);
          bf162 kk3 = *reinterpret_cast<bf162*>(&sK[kv_row * D_QK + k + 6]);
          float2 q0f = __bfloat1622float2(q0);
          float2 q1f = __bfloat1622float2(q1);
          float2 q2f = __bfloat1622float2(q2);
          float2 q3f = __bfloat1622float2(q3);
          float2 k0f = __bfloat1622float2(kk0);
          float2 k1f = __bfloat1622float2(kk1);
          float2 k2f = __bfloat1622float2(kk2);
          float2 k3f = __bfloat1622float2(kk3);
          acc += q0f.x * k0f.x + q0f.y * k0f.y
               + q1f.x * k1f.x + q1f.y * k1f.y
               + q2f.x * k2f.x + q2f.y * k2f.y
               + q3f.x * k3f.x + q3f.y * k3f.y;
        }
        sS[row * B_TOPK + warp_id * 16 + col_base + c] = acc;
      }
    }
    __syncthreads();

    // -------- Online softmax on sS, build sP (bf16) --------
    // warp 0, 16 lanes (0..15) each handle one row.
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

    // Rescale all rows of rO by scale_o_bcast[row]
    {
      const int row_per_thread = B_H;
      #pragma unroll
      for (int r = 0; r < B_H; ++r) {
        float so = scale_o_bcast[r];
        #pragma unroll
        for (int c = 0; c < 4; ++c) rO[r][c] *= so;
      }
      (void)row_per_thread;
    }

    // -------- PV: O[row, col] += sum_k P[row, k] * V[k, col] --------
    // Warp w owns N cols: warp_id * DV_PER_WARP + lane*4 + c, for c in 0..3, all 16 rows.
    {
      const int col_base_warp = warp_id * DV_PER_WARP + lane * 4;  // 0..124 within d_v
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

  // -------- Epilogue --------
  bf16* out_t = output + t * NUM_HEADS * D_V;
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
    lse_out[t * NUM_HEADS + lane] = v;
  }
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

  constexpr size_t smem_bytes = SmemLayout::TOTAL;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(
        (const void*)dsa_mla_decode_kernel_v0,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));
    attr_set = true;
  }

  const float sm_scale_log2 = static_cast<float>(sm_scale) * 1.4426950408889634f;

  dim3 grid(num_tokens);
  dim3 block(NUM_THREADS);
  dsa_mla_decode_kernel_v0<<<grid, block, smem_bytes, stream>>>(
      static_cast<const bf16*>(q_nope.data_ptr()),
      static_cast<const bf16*>(q_pe.data_ptr()),
      static_cast<const bf16*>(ckv_cache.data_ptr()),
      static_cast<const bf16*>(kpe_cache.data_ptr()),
      static_cast<const int32_t*>(sparse_indices.data_ptr()),
      static_cast<bf16*>(output.data_ptr()),
      static_cast<float*>(lse.data_ptr()),
      num_tokens, num_kv_tokens, sm_scale_log2);
}
