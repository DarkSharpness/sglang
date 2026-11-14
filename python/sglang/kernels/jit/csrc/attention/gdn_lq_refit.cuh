// LeapQuant GDN refit core (one CTA of 256 threads per value head) and the kernels built on it.
//
// GDN decays are scalar per head, so S = e^{G_l} Ds Z Dc + Ur^T diag(w) Kr in low-rank form, with
// Ur = [U~; u_0 .. u_{l-1}] [n][V], Kr = [K~; k^_0 .. k^_{l-1}] [n][K] (fp16, exact), w = [e^{G_l} x4, e^{G_l - G_j}].
// Products never materialise S: every MMA is an exact fp16 matrix (Z, Z^T, Ur, Kr) times a small fp32 factor
// split hi + lo f16 into the 8 columns of one n8 tile, with power-of-2 operand scales, fp32 accumulation.
// Warp w owns V rows [16w, 16w + 16); thread (g, t) = (lane / 4, lane % 4) holds Z rows g, g + 8, columns
// [32t, 32t + 32): the MMA K index is permuted so that this is the A fragment (k-step s, position p ->
// column 32 ((p % 8) / 2) + 4s + (p % 2) + 2 (p / 8)).
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include "gdn_lq_format.cuh"
#include <cstdint>
#include <cuda_fp16.h>

namespace sglang {

namespace lqr {

using gdn_lq::kDim;
using gdn_lq::kHeadBytes;
using gdn_lq::kRank;

SGL_DEVICE uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
SGL_DEVICE void cp_async_16_zfill(void* smem, const void* gmem, bool valid) {
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr(smem)), "l"(gmem), "r"(valid ? 16u : 0u)
      : "memory");
}
SGL_DEVICE void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n" ::: "memory");
}
SGL_DEVICE void mma_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
SGL_DEVICE uint32_t movmatrix_trans(uint32_t a) {
  uint32_t d;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(d) : "r"(a));
  return d;
}
SGL_DEVICE void ldsm_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_addr(p)));
}
SGL_DEVICE void ldsm_x4_trans(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_addr(p)));
}
SGL_DEVICE void named_bar(uint32_t id, uint32_t n) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}

/// int8x4 -> two f16x2 (bytes 0,1 -> lo; 2,3 -> hi), exact.
SGL_DEVICE void i8x4_to_f16x2(uint32_t q, uint32_t& lo, uint32_t& hi) {
  const uint32_t u = q ^ 0x80808080u;
  const uint32_t a = __byte_perm(u, 0x64646464u, 0x4140);
  const uint32_t b = __byte_perm(u, 0x64646464u, 0x4342);
  const __half2 bias = __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));
  __half2 ha = __hsub2(*reinterpret_cast<const __half2*>(&a), bias);
  __half2 hb = __hsub2(*reinterpret_cast<const __half2*>(&b), bias);
  lo = *reinterpret_cast<uint32_t*>(&ha);
  hi = *reinterpret_cast<uint32_t*>(&hb);
}
SGL_DEVICE uint32_t pack_half2(__half a, __half b) {
  __half2 h = __halves2half2(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}
SGL_DEVICE float2 half2_to_float2(uint32_t w) {
  return __half22float2(*reinterpret_cast<const __half2*>(&w));
}
SGL_DEVICE uint32_t f2h2(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);  // packed F2FP
  return *reinterpret_cast<uint32_t*>(&h);
}
/// hi (keep_lo = false) or lo (keep_lo = true) f16x2 part of (a, b).
SGL_DEVICE uint32_t hilo2(float a, float b, bool keep_lo) {
  const uint32_t hi = f2h2(a, b);
  const float2 f = half2_to_float2(hi);
  const uint32_t lo = f2h2(a - f.x, b - f.y);
  return keep_lo ? lo : hi;
}
/// Power-of-2 exponent e with max * 2^e in [2^13, 2^14) (0 for max == 0), clamped to [-126, 126].
SGL_DEVICE int op_scale_exp(float mx) {
  const int b = static_cast<int>((__float_as_uint(mx) >> 23) & 0xff);
  if (b == 0) return mx > 0.f ? 126 : 0;
  return max(-126, min(126, 13 - (b - 127)));
}
SGL_DEVICE float exp2i(int e) {
  return __int_as_float((127 + e) << 23);
}
SGL_DEVICE float warp_max_abs(float x) {
  return __uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(fabsf(x))));
}
/// e^{a - b} for fp32 a <= b <= 0 with the subtraction carried exactly (Fast2Sum).
SGL_DEVICE float exp_diff(float a, float b) {
  const float s = a - b;
  const float z = s - a;
  const float err = (-b) - z;
  return expf(s) * (1.f + err);
}

// ------------------------------------------------------------------------------------------------ refit core

#ifdef LQR_PROFILE
#define LQR_STAMP(i)                                 \
  if (stamps_ != nullptr && (threadIdx.x & 31) == 0) \
    stamps_[(blockIdx.x * 24 + (i)) * 8 + threadIdx.x / 32] = clock64();
#else
#define LQR_STAMP(i)
#endif

template <uint32_t kC, uint32_t kIters, bool kWarm>
struct RefitCore {
  using L = gdn_lq::Layout<kC>;
  static constexpr uint32_t kThreads = 256, kWarps = 8;
  static constexpr uint32_t kRows = 32;      // Ur / Kr rows: 4 compensators + kC records + zero pad
  static constexpr uint32_t kHP = kDim + 8;  // f16 pitch (272 B)
  // per-thread 32-column blocks (thread t reads [32t, 32t + 32)) padded so the 4 t-groups of a phase hit distinct banks
  static constexpr uint32_t kCP = kDim + 16;   // fp32 arrays: cpad(k) = k + 4 (k / 32)
  static constexpr uint32_t kXP = 4 * 48 + 8;  // f16 rows: xcol(k) = k + 16 (k / 32), pitch 400 B (odd 16-B units)
  SGL_DEVICE static uint32_t cpad(uint32_t k) {
    return k + 4 * (k / 32);
  }
  SGL_DEVICE static uint32_t xcol(uint32_t k) {
    return k + 16 * (k / 32);
  }
  static_assert(kRank + kC <= 24, "residual K layout assumes <= 24 low-rank rows");

  struct Smem {
    alignas(16) __half ur[kRows][kHP];  // [j][v]
    alignas(16) __half kr[kRows][kHP];  // [j][k]
    union {
      struct {
        alignas(16) float red[kWarps][2][kDim + kDim / 32][4];  // W partials [hi | lo][k][4], row pad k + k / 32
        alignas(16) float red2[kWarps][kRows][8];               // (Ur Y) partials
      } it;
      struct {
        alignas(16) __half bres[kRows][kHP];  // krp: Kr [j][permuted k], rows 24..27 = new K~
      } res;
    } u;
    alignas(16) float qs[kDim + kDim / 16][4];  // Q rows, pad k + k / 16
    alignas(16) __half xs[8][kXP];              // X = Dc Q 2^eX as [hi 0..3 | lo 4..7][xcol(k)]
    alignas(16) __half bl[8][kRows + 8];        // (w o Kr Q) 2^13 hi / lo as [n][j]
    alignas(16) __half qx[8][kXP];              // Q as [hi 0..3 | lo 4..7][xcol(k)]
    alignas(16) float c[kCP];                   // [cpad(k)]
    alignas(16) float w[kRows];
    alignas(16) float t2w[kRows][4];
    alignas(16) double gram[2][4][10];
    alignas(16) float colsum[kWarps][kCP];
    alignas(16) float rc[kCP];
    alignas(16) __half ktn[kRank][kDim];
    alignas(16) __half utn[kRank][kDim];
    float wmax[kWarps];                // per-warp max |Ur| then max |U~ new|
    float inv1[kWarps], inv2[kWarps];  // per-warp operand unscale of the W / (Ur Y) partials
    float cmax;
  };

  // per-thread state
  uint32_t zf[8][4];  // Z A fragments (k-step s)
  float srow[2];      // s of rows g, g + 8
  float eG;           // e^{G_l}
  int eX;             // X operand exponent
  uint32_t tid, lane, warp, g, t;
  int64_t* stamps_ = nullptr;

  SGL_DEVICE static uint32_t qrow(uint32_t k) {
    return k + k / 16;
  }
  SGL_DEVICE static uint32_t rrow(uint32_t k) {
    return k + k / 32;
  }
  /// column of position p in k-step s
  SGL_DEVICE static uint32_t kcol(uint32_t p, uint32_t s) {
    return 32 * ((p & 7) >> 1) + 4 * s + (p & 1) + 2 * (p >> 3);
  }

  /// Issue every load of one head. hs = this value head's region, ks = its key head's region (k^ records).
  SGL_DEVICE void load(Smem& sm, const uint8_t* hs, const uint8_t* ks, uint32_t l, const float* q0, uint32_t zw[2][8]) {
    tid = threadIdx.x;
    lane = tid % 32;
    warp = tid / 32;
    g = lane / 4;
    t = lane % 4;
    const uint32_t r0 = 16 * warp + g;
    {
      device::AlignedVector<uint32_t, 8> v0, v1;
      v0.load(hs + L::kZ + r0 * kDim + 32 * t);
      v1.load(hs + L::kZ + (r0 + 8) * kDim + 32 * t);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        zw[0][i] = v0[i], zw[1][i] = v1[i];
    }
    srow[0] = reinterpret_cast<const float*>(hs + L::kS)[r0];
    srow[1] = reinterpret_cast<const float*>(hs + L::kS)[r0 + 8];
    const uint32_t n = kRank + l;
    // Ur / Kr rows: 32 rows x 16 chunks each
#pragma unroll
    for (uint32_t it = 0; it < 2; ++it) {
      const uint32_t i = tid + it * kThreads, j = i / 16, ch = i % 16;
      const uint8_t* su = j < kRank ? hs + L::kUt + j * 256 : hs + L::kU + (j - kRank) * 256;
      const uint8_t* sk = j < kRank ? hs + L::kKt + j * 256 : ks + L::kKh + (j - kRank) * 256;
      const bool valid = j < n;
      cp_async_16_zfill(&sm.ur[j][ch * 8], valid ? su + ch * 16 : hs, valid);
      cp_async_16_zfill(&sm.kr[j][ch * 8], valid ? sk + ch * 16 : hs, valid);
    }
    if (tid < 32) {
      const float4 cv = reinterpret_cast<const float4*>(hs + L::kCc)[tid];
      *reinterpret_cast<float4*>(&sm.c[cpad(4 * tid)]) = cv;
      const float m = warp_max_abs(fmaxf(fmaxf(cv.x, cv.y), fmaxf(cv.z, cv.w)));
      if (tid == 0) sm.cmax = m;
    } else if (tid < 64) {
      const uint32_t j = tid - 32;
      const float* G = reinterpret_cast<const float*>(hs + L::kG);
      const float Gl = l > 0 ? G[l - 1] : 0.f;
      float wj = 0.f;
      if (j < kRank) {
        wj = expf(Gl);
      } else if (j < n) {
        wj = exp_diff(Gl, G[j - kRank]);
      }
      sm.w[j] = wj;
    }
    if constexpr (!kWarm) {
      if (tid >= 64 && tid < 192) {
        const uint32_t k = tid - 64;
        *reinterpret_cast<float4*>(sm.qs[qrow(k)]) = reinterpret_cast<const float4*>(q0)[k];
      }
    }
    {
      const float* G = reinterpret_cast<const float*>(hs + L::kG);
      eG = expf(l > 0 ? G[l - 1] : 0.f);
    }
  }

  /// After the loads landed: Z fragments, Z^T fragments, max |Ur|.
  SGL_DEVICE void prep(Smem& sm, const uint32_t zw[2][8]) {
#pragma unroll
    for (uint32_t s = 0; s < 8; ++s) {
      i8x4_to_f16x2(zw[0][s], zf[s][0], zf[s][2]);
      i8x4_to_f16x2(zw[1][s], zf[s][1], zf[s][3]);
    }
    // max |Ur| over the head (bounds |L| = |Ur w| since w <= 1)
    float m = 0.f;
#pragma unroll
    for (uint32_t it = 0; it < 2; ++it) {
      const uint32_t i = tid + it * kThreads, j = i / 16, ch = i % 16;
      const uint4 v = *reinterpret_cast<const uint4*>(&sm.ur[j][ch * 8]);
      const uint32_t wv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 f = half2_to_float2(wv[q]);
        m = fmaxf(m, fmaxf(fabsf(f.x), fabsf(f.y)));
      }
    }
    m = warp_max_abs(m);
    if (lane == 0) sm.wmax[warp] = m;
    eX = op_scale_exp(sm.cmax);
  }

  /// Q (threads 0..127, row k) -> qs and the X = Dc Q 2^eX operand.
  SGL_DEVICE void put_q(Smem& sm, uint32_t k, const float (&q)[4]) {
    *reinterpret_cast<float4*>(sm.qs[qrow(k)]) = make_float4(q[0], q[1], q[2], q[3]);
    const float sc = sm.c[cpad(k)] * exp2i(eX);
#pragma unroll
    for (int j = 0; j < 4; j += 2) {
      const float x0 = q[j] * sc, x1 = q[j + 1] * sc;
      const uint32_t hi = hilo2(x0, x1, false), lo = hilo2(x0, x1, true);
      sm.xs[j][xcol(k)] = __ushort_as_half(static_cast<unsigned short>(hi));
      sm.xs[j + 1][xcol(k)] = __ushort_as_half(static_cast<unsigned short>(hi >> 16));
      sm.xs[4 + j][xcol(k)] = __ushort_as_half(static_cast<unsigned short>(lo));
      sm.xs[5 + j][xcol(k)] = __ushort_as_half(static_cast<unsigned short>(lo >> 16));
      const uint32_t qh = hilo2(q[j], q[j + 1], false), ql = hilo2(q[j], q[j + 1], true);
      sm.qx[j][xcol(k)] = __ushort_as_half(static_cast<unsigned short>(qh));
      sm.qx[j + 1][xcol(k)] = __ushort_as_half(static_cast<unsigned short>(qh >> 16));
      sm.qx[4 + j][xcol(k)] = __ushort_as_half(static_cast<unsigned short>(ql));
      sm.qx[5 + j][xcol(k)] = __ushort_as_half(static_cast<unsigned short>(ql >> 16));
    }
  }

  /// T1 = Kr Q, B operand (w o T1) 2^13 hi / lo of the low-rank Y product. All threads.
  SGL_DEVICE void low_rank_t1(Smem& sm) {
    // T1 = Kr Q on warps 0, 1 (m-tile = 16 rows of Kr each), B = Q hi / lo in the permuted K order of xs
    if (warp >= 2) return;
    uint32_t ka[2][16], qb[16];
#pragma unroll
    for (uint32_t h = 0; h < 2; ++h) {
      const uint4* kp = reinterpret_cast<const uint4*>(&sm.kr[16 * warp + g + 8 * h][32 * t]);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 v = kp[i];
        ka[h][4 * i] = v.x, ka[h][4 * i + 1] = v.y, ka[h][4 * i + 2] = v.z, ka[h][4 * i + 3] = v.w;
      }
    }
    {
      const uint4* qp = reinterpret_cast<const uint4*>(&sm.qx[g][xcol(32 * t)]);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 v = qp[i];
        qb[4 * i] = v.x, qb[4 * i + 1] = v.y, qb[4 * i + 2] = v.z, qb[4 * i + 3] = v.w;
      }
    }
    float a0[4] = {0.f, 0.f, 0.f, 0.f}, a1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (uint32_t s = 0; s < 8; s += 2) {
      const uint32_t fa[4] = {ka[0][2 * s], ka[1][2 * s], ka[0][2 * s + 1], ka[1][2 * s + 1]};
      const uint32_t fb[4] = {ka[0][2 * s + 2], ka[1][2 * s + 2], ka[0][2 * s + 3], ka[1][2 * s + 3]};
      mma_16816(a0, fa, qb[2 * s], qb[2 * s + 1]);
      mma_16816(a1, fb, qb[2 * s + 2], qb[2 * s + 3]);
    }
    float v[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const float p = a0[r] + a1[r];
      v[r] = p + __shfl_xor_sync(0xffffffffu, p, 2);  // hi + lo columns: T1[j][2 (t % 2) + (r % 2)]
    }
    const uint32_t j0 = 16 * warp + g;
    const float w0 = sm.w[j0] * 8192.f, w1 = sm.w[j0 + 8] * 8192.f;
    const bool lo = t >= 2;
    const uint32_t h0 = hilo2(v[0] * w0, v[1] * w0, lo), h1 = hilo2(v[2] * w1, v[3] * w1, lo);
    sm.bl[2 * t][j0] = __ushort_as_half(static_cast<unsigned short>(h0));
    sm.bl[2 * t + 1][j0] = __ushort_as_half(static_cast<unsigned short>(h0 >> 16));
    sm.bl[2 * t][j0 + 8] = __ushort_as_half(static_cast<unsigned short>(h1));
    sm.bl[2 * t + 1][j0 + 8] = __ushort_as_half(static_cast<unsigned short>(h1 >> 16));
  }

  /// Y = S Q for the warp's rows: y[r] = Y[v][j], r: (g, j0), (g, j0 + 1), (g + 8, j0), (g + 8, j0 + 1),
  /// j0 = 2 (t % 2) (threads t and t ^ 2 hold the same values).
  SGL_DEVICE void product_y(Smem& sm, float (&y)[4]) {
    uint32_t xb[16];
    {
      const uint4* xp = reinterpret_cast<const uint4*>(&sm.xs[g][xcol(32 * t)]);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 v = xp[i];
        xb[4 * i] = v.x, xb[4 * i + 1] = v.y, xb[4 * i + 2] = v.z, xb[4 * i + 3] = v.w;
      }
    }
    float a0[4] = {0.f, 0.f, 0.f, 0.f}, a1[4] = {0.f, 0.f, 0.f, 0.f}, al[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (uint32_t s = 0; s < 8; s += 2) {
      mma_16816(a0, zf[s], xb[2 * s], xb[2 * s + 1]);
      mma_16816(a1, zf[s + 1], xb[2 * s + 2], xb[2 * s + 3]);
    }
#pragma unroll
    for (uint32_t q = 0; q < 2; ++q) {
      const uint32_t m = lane >> 3;
      uint32_t ua[4];
      ldsm_x4_trans(ua, &sm.ur[16 * q + (lane & 7) + 8 * (m >> 1)][16 * warp + 8 * (m & 1)]);
      const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sm.bl[g][16 * q + 2 * t]);
      const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sm.bl[g][16 * q + 2 * t + 8]);
      mma_16816(al, ua, b0, b1);
    }
    const float zs0 = eG * srow[0] * exp2i(-eX), zs1 = eG * srow[1] * exp2i(-eX);
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const float p = (a0[r] + a1[r]) * (r < 2 ? zs0 : zs1) + al[r] * (1.f / 8192.f);
      y[r] = p + __shfl_xor_sync(0xffffffffu, p, 2);
    }
  }

  /// B fragments (kk = own V rows, n = [hi | lo]) of x (C layout, values of Y-type) via movmatrix; returns 2^-e.
  SGL_DEVICE float b_from_c(const float (&x)[4], uint32_t& b0, uint32_t& b1) {
    float m = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
    m = warp_max_abs(m);
    const int e = op_scale_exp(m);
    const float sc = exp2i(e);
    const bool lo = t >= 2;
    const uint32_t p0 = hilo2(x[0] * sc, x[1] * sc, lo);
    const uint32_t p1 = hilo2(x[2] * sc, x[3] * sc, lo);
    b0 = movmatrix_trans(p0);
    b1 = movmatrix_trans(p1);
    return exp2i(-e);
  }

  /// Per-warp partials of Z^T Ds Y and Ur Y -> smem (after this: __syncthreads, then reduce_w).
  SGL_DEVICE void product_w_partial(Smem& sm, const float (&y)[4]) {
    uint32_t bw0, bw1, bt0, bt1;
    const float xs1[4] = {y[0] * srow[0], y[1] * srow[0], y[2] * srow[1], y[3] * srow[1]};
    const float inv1 = b_from_c(xs1, bw0, bw1);
    const float inv2 = b_from_c(y, bt0, bt1);
    float aw[8][4], at[2][4];
#pragma unroll
    for (uint32_t s = 0; s < 8; ++s) {
      // Z^T tile = transposes of the own Z tile: a'0 = T(rows g, pos 0-7), a'1 = T(rows g, pos 8-15),
      // a'2 = T(rows g+8, pos 0-7), a'3 = T(rows g+8, pos 8-15)
      const uint32_t zt[4] = {
          movmatrix_trans(zf[s][0]), movmatrix_trans(zf[s][2]), movmatrix_trans(zf[s][1]), movmatrix_trans(zf[s][3])};
      aw[s][0] = aw[s][1] = aw[s][2] = aw[s][3] = 0.f;
      mma_16816(aw[s], zt, bw0, bw1);
    }
#pragma unroll
    for (uint32_t mt = 0; mt < 2; ++mt) {
      const uint32_t m = lane >> 3;
      uint32_t ua[4];
      ldsm_x4(ua, &sm.ur[16 * mt + (lane & 7) + 8 * (m & 1)][16 * warp + 8 * (m >> 1)]);
      at[mt][0] = at[mt][1] = at[mt][2] = at[mt][3] = 0.f;
      mma_16816(at[mt], ua, bt0, bt1);
    }
    if (lane == 0) sm.inv1[warp] = inv1, sm.inv2[warp] = inv2;
#pragma unroll
    for (uint32_t s = 0; s < 8; ++s) {
      const uint32_t k0 = kcol(g, s), k1 = kcol(g + 8, s);
      *reinterpret_cast<float2*>(&sm.u.it.red[warp][t >> 1][rrow(k0)][2 * (t & 1)]) = make_float2(aw[s][0], aw[s][1]);
      *reinterpret_cast<float2*>(&sm.u.it.red[warp][t >> 1][rrow(k1)][2 * (t & 1)]) = make_float2(aw[s][2], aw[s][3]);
    }
#pragma unroll
    for (uint32_t mt = 0; mt < 2; ++mt) {
      *reinterpret_cast<float2*>(&sm.u.it.red2[warp][16 * mt + g][2 * t]) = make_float2(at[mt][0], at[mt][1]);
      *reinterpret_cast<float2*>(&sm.u.it.red2[warp][16 * mt + g + 8][2 * t]) = make_float2(at[mt][2], at[mt][3]);
    }
  }

  /// Threads 0..127: W base row k (sum over warps); threads 128..255: t2w = w o (Ur Y).
  SGL_DEVICE void reduce_w(Smem& sm, float (&wb)[4]) {
    if (tid < kDim) {
      const uint32_t k = tid;
      float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (uint32_t w = 0; w < kWarps; ++w) {
        const float4 h = *reinterpret_cast<const float4*>(&sm.u.it.red[w][0][rrow(k)][0]);
        const float4 l = *reinterpret_cast<const float4*>(&sm.u.it.red[w][1][rrow(k)][0]);
        const float sc = sm.inv1[w];
        a.x += (h.x + l.x) * sc, a.y += (h.y + l.y) * sc, a.z += (h.z + l.z) * sc, a.w += (h.w + l.w) * sc;
      }
      wb[0] = a.x, wb[1] = a.y, wb[2] = a.z, wb[3] = a.w;
    } else {
      const uint32_t i = (tid - kDim) >> 2, j = tid & 3;
      float a = 0.f;
#pragma unroll
      for (uint32_t w = 0; w < kWarps; ++w)
        a += (sm.u.it.red2[w][i][j] + sm.u.it.red2[w][i][j + 4]) * sm.inv2[w];
      sm.t2w[i][j] = a * sm.w[i];
    }
  }

  /// Threads 0..127: W row k = e^{G_l} c_k wb + Kr^T (w o Ur Y).
  SGL_DEVICE void finish_w(Smem& sm, uint32_t k, const float (&wb)[4], float (&wr)[4]) {
    const float ck = eG * sm.c[cpad(k)];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      wr[j] = ck * wb[j];
#pragma unroll
    for (uint32_t i = 0; i < 24; ++i) {
      const float kv = __half2float(sm.kr[i][k]);
      const float4 tv = *reinterpret_cast<const float4*>(sm.t2w[i]);
      wr[0] += kv * tv.x, wr[1] += kv * tv.y, wr[2] += kv * tv.z, wr[3] += kv * tv.w;
    }
  }

  /// One Cholesky-QR pass on the 128 x 4 block held one row per thread (threads 0..127), fp64 Gram.
  template <typename T>
  SGL_DEVICE void cholqr_pass(Smem& sm, T (&q)[4], uint32_t buf) {
    T p[16];
    p[0] = q[0] * q[0], p[1] = q[0] * q[1], p[2] = q[0] * q[2], p[3] = q[0] * q[3];
    p[4] = q[1] * q[1], p[5] = q[1] * q[2], p[6] = q[1] * q[3];
    p[7] = q[2] * q[2], p[8] = q[2] * q[3], p[9] = q[3] * q[3];
#pragma unroll
    for (int i = 10; i < 16; ++i)
      p[i] = T(0);
    // transposed butterfly: lane ends with the total of entry lane / 2
#pragma unroll
    for (uint32_t half = 8, off = 16; half >= 1; half >>= 1, off >>= 1) {
      const bool up = (lane & off) != 0;
#pragma unroll
      for (uint32_t i = 0; i < half; ++i) {
        const T send = up ? p[i] : p[i + half];
        const T keep = up ? p[i + half] : p[i];
        p[i] = keep + __shfl_xor_sync(0xffffffffu, send, off);
      }
    }
    p[0] += __shfl_xor_sync(0xffffffffu, p[0], 1);
    if (buf == 0) LQR_STAMP(14);
    if ((lane & 1) == 0 && (lane >> 1) < 10) sm.gram[buf][warp][lane >> 1] = p[0];
    named_bar(1, kDim);
    if (buf == 0) LQR_STAMP(15);
    T G[10];
#pragma unroll
    for (int i = 0; i < 10; ++i)
      G[i] = (T)sm.gram[buf][0][i] + sm.gram[buf][1][i] + sm.gram[buf][2][i] + sm.gram[buf][3][i];
    // G = R^T R, R upper triangular; dead columns (pivot ~ 0) give a zero column
    const auto gi = [](int a, int b) { return a == 0 ? b : a == 1 ? 3 + b : a == 2 ? 5 + b : 9; };
    T r[4][4], rinv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int i = 0; i < j; ++i) {
        T v = G[gi(i, j)];
#pragma unroll
        for (int m = 0; m < i; ++m)
          v -= r[m][i] * r[m][j];
        r[i][j] = v * rinv[i];
      }
      T d = G[gi(j, j)];
#pragma unroll
      for (int m = 0; m < j; ++m)
        d -= r[m][j] * r[m][j];
      const bool live = d > T(sizeof(T) == 8 ? 1e-30 : 1e-12) * G[gi(j, j)] && G[gi(j, j)] > T(0);
      T y = static_cast<T>(rsqrtf(static_cast<float>(d)));
      if constexpr (sizeof(T) == 8) {
        y = y * (1.5 - 0.5 * d * y * y);
        y = y * (1.5 - 0.5 * d * y * y);
      }
      rinv[j] = live ? y : T(0);
    }
    if (buf == 0) LQR_STAMP(16);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      T v = q[j];
#pragma unroll
      for (int i = 0; i < j; ++i)
        v -= q[i] * r[i][j];
      q[j] = v * rinv[j];
    }
  }

  /// Threads 0..127: orthonormalise row k of W (CholeskyQR2) and publish Q.
  SGL_DEVICE void orth_put(Smem& sm, uint32_t k, const float (&wr)[4], bool last) {
#ifdef LQR_ORTH32
    using OrthT = float;
#else
    using OrthT = double;
#endif
    OrthT q[4] = {wr[0], wr[1], wr[2], wr[3]};
    LQR_STAMP(22);
    cholqr_pass(sm, q, 0);
    LQR_STAMP(23);
#ifdef LQR_CHOL2
    cholqr_pass(sm, q, 1);
#endif
    float qf[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      qf[j] = static_cast<float>(q[j]);
    if (last) {
      // K~ = fp16(Q^T); the U~ product uses the rounded K~
#pragma unroll
      for (int j = 0; j < 4; j += 2) {
        const uint32_t h = f2h2(qf[j], qf[j + 1]);
        sm.ktn[j][k] = __ushort_as_half(static_cast<unsigned short>(h));
        sm.ktn[j + 1][k] = __ushort_as_half(static_cast<unsigned short>(h >> 16));
        const float2 f = half2_to_float2(h);
        qf[j] = f.x, qf[j + 1] = f.y;
      }
    }
    put_q(sm, k, qf);
  }

  /// Full refit of one head; writes Z, s, c, K~, U~ of the head region hs_out (may equal hs).
  SGL_DEVICE void
  run(Smem& sm, const uint8_t* hs, const uint8_t* ks, uint8_t* hs_out, uint32_t l, const float* q0, float floor_c) {
    uint32_t zw[2][8];
    LQR_STAMP(0);
    load(sm, hs, ks, l, q0, zw);
    cp_async_wait_all();
    __syncthreads();
    LQR_STAMP(1);
    prep(sm, zw);
    if (tid < kDim) {
      float q[4];
      if constexpr (kWarm) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
          q[j] = __half2float(sm.kr[j][tid]);
      } else {
        const float4 v = *reinterpret_cast<const float4*>(sm.qs[qrow(tid)]);
        q[0] = v.x, q[1] = v.y, q[2] = v.z, q[3] = v.w;
      }
      put_q(sm, tid, q);
    }
    __syncthreads();
    LQR_STAMP(2);
#pragma unroll 1
    for (uint32_t it = 0; it <= kIters; ++it) {
      low_rank_t1(sm);
      __syncthreads();
      LQR_STAMP(3 + 5 * it);
      float y[4];
      product_y(sm, y);
      LQR_STAMP(4 + 5 * it);
      if (it == kIters) {
        // U~ = fp16(S K~^T)^T
        float m = 0.f;
        if (t < 2) {
          const uint32_t v0 = 16 * warp + g;
          const uint32_t ha = f2h2(y[0], y[1]), hb = f2h2(y[2], y[3]);
          const __half h0 = __ushort_as_half(static_cast<unsigned short>(ha)),
                       h1 = __ushort_as_half(static_cast<unsigned short>(ha >> 16));
          const __half h2 = __ushort_as_half(static_cast<unsigned short>(hb)),
                       h3 = __ushort_as_half(static_cast<unsigned short>(hb >> 16));
          sm.utn[2 * t][v0] = h0, sm.utn[2 * t + 1][v0] = h1;
          sm.utn[2 * t][v0 + 8] = h2, sm.utn[2 * t + 1][v0 + 8] = h3;
          m = fmaxf(
              fmaxf(fabsf(__half2float(h0)), fabsf(__half2float(h1))),
              fmaxf(fabsf(__half2float(h2)), fabsf(__half2float(h3))));
        }
        m = warp_max_abs(m);
        // wmax holds max |Ur| per warp from prep; fold the U~ max in (read after the barrier)
        if (lane == 0) sm.wmax[warp] = fmaxf(sm.wmax[warp], m);
        break;
      }
      product_w_partial(sm, y);
      __syncthreads();
      LQR_STAMP(5 + 5 * it);
      float wb[4];
      reduce_w(sm, wb);
      __syncthreads();
      LQR_STAMP(6 + 5 * it);
      if (tid < kDim) {
        float wr[4];
        finish_w(sm, tid, wb, wr);
        orth_put(sm, tid, wr, it + 1 == kIters);
      }
      __syncthreads();
      LQR_STAMP(7 + 5 * it);
    }
    __syncthreads();
    LQR_STAMP(17);
    residual(sm, hs_out, floor_c);
  }

  /// Residual R = S - U~^T K~ as MMA accumulators, then c, s, Z; stores the new base.
  /// K layout: k-step 0 = L hi rows 0..15, 1 = L hi rows 16..23 | -U~ (kk 24..27), 2 = L lo 0..15, 3 = L lo 16..23,
  /// against B rows krp 0..15 / 16..31 (krp = Kr in the permuted column order, rows 24..27 = new K~).
  SGL_DEVICE void residual(Smem& sm, uint8_t* hs_out, float floor_c) {
    namespace math2 = device::math;
    float mx = 0.f;
#pragma unroll
    for (uint32_t w = 0; w < kWarps; ++w)
      mx = fmaxf(mx, sm.wmax[w]);
    const int eL = op_scale_exp(mx);
    const float sL = exp2i(eL);
    // krp: position pair vp = 4 nt + a' <- natural word 16 a' + nt
#pragma unroll
    for (uint32_t it = 0; it < kRows * kDim / 2 / kThreads; ++it) {
      const uint32_t i = tid + it * kThreads, row = i / (kDim / 2), vp = i % (kDim / 2);
      const uint32_t nw = 16 * (vp & 3) + (vp >> 2);
      const bool kt = row >= 24 && row < 24 + kRank;
      const uint32_t b = kt ? *reinterpret_cast<const uint32_t*>(&sm.ktn[row - 24][2 * nw])
                            : *reinterpret_cast<const uint32_t*>(&sm.kr[row][2 * nw]);
      *reinterpret_cast<uint32_t*>(&sm.u.res.bres[row][2 * vp]) = b;
    }
    // A fragments of the own V rows, in registers
    uint32_t ah[2][4], al[2][4];
    {
      const uint32_t m = lane >> 3;
#pragma unroll
      for (uint32_t q = 0; q < 2; ++q) {
        uint32_t u[4];
        ldsm_x4_trans(u, &sm.ur[16 * q + (lane & 7) + 8 * (m >> 1)][16 * warp + 8 * (m & 1)]);
        const float2 w01 = *reinterpret_cast<const float2*>(&sm.w[16 * q + 2 * t]);
        const float2 w89 = *reinterpret_cast<const float2*>(&sm.w[16 * q + 2 * t + 8]);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const float2 wv = r < 2 ? w01 : w89;
          const float2 f = half2_to_float2(u[r]);
          const float x0 = f.x * wv.x * sL, x1 = f.y * wv.y * sL;
          ah[q][r] = hilo2(x0, x1, false);
          al[q][r] = hilo2(x0, x1, true);
        }
      }
      // k-step 1 positions 8..15 = kk 24..31: -U~ rows 0..3 (t < 2), zero otherwise; k-step 3 positions 8..15 = 0
      uint32_t ut0 = 0, ut1 = 0;
      if (t < 2) {
        const uint32_t v0 = 16 * warp + g;
        ut0 = f2h2(-__half2float(sm.utn[2 * t][v0]) * sL, -__half2float(sm.utn[2 * t + 1][v0]) * sL);
        ut1 = f2h2(-__half2float(sm.utn[2 * t][v0 + 8]) * sL, -__half2float(sm.utn[2 * t + 1][v0 + 8]) * sL);
      }
      ah[1][2] = ut0, ah[1][3] = ut1;
      al[1][2] = 0u, al[1][3] = 0u;
    }
    // new K~ / U~ to global (1 KB each)
    if (tid < 64) {
      reinterpret_cast<uint4*>(hs_out + L::kKt)[tid] = reinterpret_cast<const uint4*>(&sm.ktn[0][0])[tid];
    } else if (tid < 128) {
      reinterpret_cast<uint4*>(hs_out + L::kUt)[tid - 64] = reinterpret_cast<const uint4*>(&sm.utn[0][0])[tid - 64];
    }
    __syncthreads();
    LQR_STAMP(18);
    float2 acc[16][2];  // [nt][row g / g + 8] = columns (32t + 2nt, +1)
    {
      const float2 b0 = make_float2(sL * eG * srow[0], sL * eG * srow[0]);
      const float2 b1 = make_float2(sL * eG * srow[1], sL * eG * srow[1]);
#pragma unroll
      for (uint32_t nt = 0; nt < 16; ++nt) {
        const float2 cv = *reinterpret_cast<const float2*>(&sm.c[cpad(32 * t) + 2 * nt]);
        const uint32_t s = nt >> 1;
        const float2 z0 = half2_to_float2((nt & 1) ? zf[s][2] : zf[s][0]);
        const float2 z1 = half2_to_float2((nt & 1) ? zf[s][3] : zf[s][1]);
        acc[nt][0] = math2::mul_f32x2(math2::mul_f32x2(b0, cv), z0);
        acc[nt][1] = math2::mul_f32x2(math2::mul_f32x2(b1, cv), z1);
      }
    }
    {
      const uint32_t m = lane >> 3;
#pragma unroll
      for (uint32_t q = 0; q < 2; ++q) {
#pragma unroll
        for (uint32_t nt = 0; nt < 16; nt += 2) {
          uint32_t b[4];
          ldsm_x4_trans(b, &sm.u.res.bres[16 * q + (lane & 7) + 8 * (m & 1)][8 * (nt + (m >> 1))]);
          float (&c0)[4] = *reinterpret_cast<float (*)[4]>(&acc[nt][0]);
          float (&c1)[4] = *reinterpret_cast<float (*)[4]>(&acc[nt + 1][0]);
          mma_16816(c0, ah[q], b[0], b[1]);
          mma_16816(c1, ah[q], b[2], b[3]);
          mma_16816(c0, al[q], b[0], b[1]);
          mma_16816(c1, al[q], b[2], b[3]);
        }
      }
    }
    LQR_STAMP(19);
    // column sums of |R| 2^eL: own 2 rows, then transposed butterfly over g (lane bits 2..4)
    {
      float cs[32];
#pragma unroll
      for (uint32_t nt = 0; nt < 16; ++nt) {
        cs[2 * nt] = fabsf(acc[nt][0].x) + fabsf(acc[nt][1].x);
        cs[2 * nt + 1] = fabsf(acc[nt][0].y) + fabsf(acc[nt][1].y);
      }
#pragma unroll
      for (uint32_t half = 16, off = 16; half >= 4; half >>= 1, off >>= 1) {
        const bool up = (lane & off) != 0;
#pragma unroll
        for (uint32_t i = 0; i < half; ++i) {
          const float send = up ? cs[i] : cs[i + half];
          const float keep = up ? cs[i + half] : cs[i];
          cs[i] = keep + __shfl_xor_sync(0xffffffffu, send, off);
        }
      }
      // lane holds columns 32t + 4g + 0..3
      *reinterpret_cast<float4*>(&sm.colsum[warp][cpad(32 * t) + 4 * g]) = make_float4(cs[0], cs[1], cs[2], cs[3]);
    }
    __syncthreads();
    LQR_STAMP(20);
    const float sLinv = exp2i(-eL);
    if (tid < kDim) {
      float a = 0.f;
#pragma unroll
      for (uint32_t w = 0; w < kWarps; ++w)
        a += sm.colsum[w][cpad(tid)];
      const float cnew = fmaxf(sqrtf(a * sLinv * (1.f / kDim)), floor_c);
      reinterpret_cast<float*>(hs_out + L::kCc)[tid] = cnew;
      sm.rc[cpad(tid)] = __fdiv_rn(sLinv, cnew);
    }
    __syncthreads();
    float rmax0 = 0.f, rmax1 = 0.f;
#pragma unroll
    for (uint32_t nt = 0; nt < 16; ++nt) {
      const float2 r = *reinterpret_cast<const float2*>(&sm.rc[cpad(32 * t) + 2 * nt]);
      acc[nt][0] = math2::mul_f32x2(acc[nt][0], r);
      acc[nt][1] = math2::mul_f32x2(acc[nt][1], r);
      rmax0 = fmaxf(rmax0, fmaxf(fabsf(acc[nt][0].x), fabsf(acc[nt][0].y)));
      rmax1 = fmaxf(rmax1, fmaxf(fabsf(acc[nt][1].x), fabsf(acc[nt][1].y)));
    }
#pragma unroll
    for (uint32_t m = 1; m < 4; m <<= 1) {
      rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, m));
      rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, m));
    }
    float s0 = __fdiv_rn(rmax0, 127.f), s1 = __fdiv_rn(rmax1, 127.f);
    s0 = s0 == 0.f ? 1.f : s0;
    s1 = s1 == 0.f ? 1.f : s1;
    const uint32_t r0 = 16 * warp + g;
    if (t == 0) {
      reinterpret_cast<float*>(hs_out + L::kS)[r0] = s0;
      reinterpret_cast<float*>(hs_out + L::kS)[r0 + 8] = s1;
    }
    // |R' / s| <= 127 (1 + a few ulp) by construction, so the rounded value needs no clamp
    // x * inv + 1.5 * 2^23 rounds x to the nearest integer (ties to even) into the low mantissa bits
    const float2 i0 = make_float2(__frcp_rn(s0), __frcp_rn(s0)), i1 = make_float2(__frcp_rn(s1), __frcp_rn(s1));
    const float2 mg = make_float2(12582912.f, 12582912.f);
    device::AlignedVector<uint32_t, 8> o0, o1;
    const auto bits = [](float x) { return __float_as_uint(x); };
#pragma unroll
    for (uint32_t wd = 0; wd < 8; ++wd) {
      const float2 a0 = math2::fma_f32x2(acc[2 * wd][0], i0, mg), a1 = math2::fma_f32x2(acc[2 * wd + 1][0], i0, mg);
      const float2 b0 = math2::fma_f32x2(acc[2 * wd][1], i1, mg), b1 = math2::fma_f32x2(acc[2 * wd + 1][1], i1, mg);
      o0[wd] =
          __byte_perm(__byte_perm(bits(a0.x), bits(a0.y), 0x0040), __byte_perm(bits(a1.x), bits(a1.y), 0x0040), 0x5410);
      o1[wd] =
          __byte_perm(__byte_perm(bits(b0.x), bits(b0.y), 0x0040), __byte_perm(bits(b1.x), bits(b1.y), 0x0040), 0x5410);
    }
    o0.store(hs_out + L::kZ + r0 * kDim + 32 * t);
    o1.store(hs_out + L::kZ + (r0 + 8) * kDim + 32 * t);
    LQR_STAMP(21);
  }
};

struct RefitListParams {
  uint8_t* __restrict__ pool;         // [layers][slots][HV][kHeadBytes]
  const int32_t* __restrict__ layer;  // [M]
  const int32_t* __restrict__ slot;   // [M]
  const int32_t* __restrict__ lnew;   // [M] window length at the refit
  const float* __restrict__ q0;       // [K][4]
  int64_t layer_stride, slot_stride;  // bytes
  uint32_t num_heads;
  float floor_c;
  int64_t* stamps;  // profile builds only
};

}  // namespace lqr

/// Test entry: refit every (item, head) in place; grid = items x heads.
template <uint32_t kC, uint32_t kIters, bool kWarm>
__global__ void __launch_bounds__(256, 2) gdn_lq_refit_list_kernel(const __grid_constant__ lqr::RefitListParams p) {
  using Core = lqr::RefitCore<kC, kIters, kWarm>;
  extern __shared__ __align__(16) uint8_t smem_raw[];
  auto& sm = *reinterpret_cast<typename Core::Smem*>(smem_raw);
  const uint32_t item = blockIdx.x / p.num_heads, h = blockIdx.x % p.num_heads;
  uint8_t* base = p.pool + p.layer[item] * p.layer_stride + p.slot[item] * p.slot_stride;
  uint8_t* hs = base + static_cast<int64_t>(h) * lqr::kHeadBytes;
  const uint8_t* ks = base + static_cast<int64_t>(h & ~1u) * lqr::kHeadBytes;
  Core core;
  core.stamps_ = p.stamps;
  core.run(sm, hs, ks, hs, static_cast<uint32_t>(p.lnew[item]), p.q0, p.floor_c);
}

template <uint32_t kC, uint32_t kIters, bool kWarm>
void gdn_lq_refit_list(
    tvm::ffi::TensorView pool,
    tvm::ffi::TensorView layer,
    tvm::ffi::TensorView slot,
    tvm::ffi::TensorView lnew,
    tvm::ffi::TensorView q0,
    double floor_c,
    int64_t stamps) {
  using namespace host;
  using Core = lqr::RefitCore<kC, kIters, kWarm>;
  auto NL = SymbolicSize{"layers"}, NS = SymbolicSize{"slots"}, HV = SymbolicSize{"heads"}, M = SymbolicSize{"items"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({NL, NS, HV, 128, 128})  //
      .with_dtype<float>()
      .with_device<kDLCUDA>(device)
      .verify(pool);
  TensorMatcher({M})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device)
      .verify(layer)
      .verify(slot)
      .verify(lnew);
  TensorMatcher({128, 4})  //
      .with_dtype<float>()
      .with_device<kDLCUDA>(device)
      .verify(q0);
  const uint32_t m = static_cast<uint32_t>(M.unwrap()), hv = static_cast<uint32_t>(HV.unwrap());
  CHECK_HOST(hv % 2 == 0) << "value heads must pair with key heads";
  if (m == 0) return;
  lqr::RefitListParams p{};
  p.pool = static_cast<uint8_t*>(pool.data_ptr());
  p.layer = static_cast<const int32_t*>(layer.data_ptr());
  p.slot = static_cast<const int32_t*>(slot.data_ptr());
  p.lnew = static_cast<const int32_t*>(lnew.data_ptr());
  p.q0 = static_cast<const float*>(q0.data_ptr());
  p.slot_stride = static_cast<int64_t>(hv) * lqr::kHeadBytes;
  p.layer_stride = NS.unwrap() * p.slot_stride;
  p.num_heads = hv;
  p.floor_c = static_cast<float>(floor_c);
  p.stamps = reinterpret_cast<int64_t*>(stamps);
  constexpr uint32_t kSmem = sizeof(typename Core::Smem);
  runtime::set_max_dynamic_smem_per_device<gdn_lq_refit_list_kernel<kC, kIters, kWarm>>(device.unwrap(), kSmem);
  LaunchKernel(m * hv, Core::kThreads, device.unwrap(), kSmem)(gdn_lq_refit_list_kernel<kC, kIters, kWarm>, p);
}

// ------------------------------------------------------------------------------------------------ commit / flush
// Step commit for every (layer, request) pair p = layer * N + i: l_new = l + n_i (n = 0 for a decode flush); the pair
// is due when l_new >= lim (lim 0 = P). Kernel 1 (one CTA) computes the new (l, lim) of every pair and compacts the
// due pairs; kernel 2 (persistent) writes l / lim to every value-head copy, then refits every head of the due pairs.

namespace lqr {

struct CommitParams {
  uint8_t* __restrict__ pool;               // [layers][slots][HV][kHeadBytes]
  const int32_t* __restrict__ slot_idx;     // [N], < 0 = padding
  const int32_t* __restrict__ n_acc;        // [N] accepted tokens, nullptr = 0 (flush)
  int32_t* __restrict__ pair_l;             // [L * N] new l (-1: padding row)
  int32_t* __restrict__ pair_lim;           // [L * N] new lim
  int32_t* __restrict__ pair_lnew;          // [L * N] l at the refit (due pairs)
  int32_t* __restrict__ due_list;           // [L * N]
  int32_t* __restrict__ due_count;          // [1]
  const int32_t* __restrict__ track_idx;    // [N] track slot (< 0: none), nullptr = no tracking
  const int32_t* __restrict__ track_steps;  // [N] accepted-step index to snapshot (< 0: none)
  int32_t* __restrict__ track_list;         // [L * N] pairs to snapshot
  int32_t* __restrict__ track_l;            // [L * N] l of the snapshot
  int32_t* __restrict__ track_count;        // [1]
  const float* __restrict__ q0;
  int64_t layer_stride, slot_stride;
  uint32_t num_layers, num_req, num_heads, lofs;  // lofs: byte offset of l in the head region
  int32_t P;
  float floor_c;
};

}  // namespace lqr

__global__ void __launch_bounds__(1024, 1) gdn_lq_commit_scan_kernel(const __grid_constant__ lqr::CommitParams p) {
  __shared__ uint32_t count, tcount;
  if (threadIdx.x == 0) count = 0, tcount = 0;
  __syncthreads();
  const uint32_t pairs = p.num_layers * p.num_req;
  for (uint32_t base = 0; base < pairs; base += blockDim.x) {
    const uint32_t q = base + threadIdx.x;
    bool due = false, track = false;
    if (q < pairs) {
      const uint32_t layer = q / p.num_req, i = q % p.num_req;
      const int32_t slot = p.slot_idx[i];
      if (slot >= 0) {
        const int2 ll = *reinterpret_cast<const int2*>(p.pool + layer * p.layer_stride + slot * p.slot_stride + p.lofs);
        if (p.track_idx != nullptr) {
          const int32_t ts = p.track_steps[i];
          track = ts >= 0 && p.track_idx[i] >= 0;
          if (track) p.track_l[q] = ll.x + ts + 1;  // the snapshot keeps the records of the accepted prefix
        }
        const int32_t lnew = ll.x + (p.n_acc != nullptr ? p.n_acc[i] : 0);
        const int32_t lim = ll.y > 0 ? ll.y : p.P;
        due = lnew >= lim;
        p.pair_l[q] = due ? 0 : lnew;
        p.pair_lim[q] = due ? p.P : lim;
        p.pair_lnew[q] = lnew;
      } else {
        p.pair_l[q] = -1;
      }
    }
    const uint32_t mask = __ballot_sync(0xffffffffu, due);
    uint32_t wbase = 0;
    if ((threadIdx.x & 31) == 0 && mask != 0) wbase = atomicAdd(&count, __popc(mask));
    wbase = __shfl_sync(0xffffffffu, wbase, 0);
    if (due) p.due_list[wbase + __popc(mask & ((1u << (threadIdx.x & 31)) - 1))] = static_cast<int32_t>(q);
    const uint32_t tmask = __ballot_sync(0xffffffffu, track);
    uint32_t tbase = 0;
    if ((threadIdx.x & 31) == 0 && tmask != 0) tbase = atomicAdd(&tcount, __popc(tmask));
    tbase = __shfl_sync(0xffffffffu, tbase, 0);
    if (track) p.track_list[tbase + __popc(tmask & ((1u << (threadIdx.x & 31)) - 1))] = static_cast<int32_t>(q);
  }
  __syncthreads();
  if (threadIdx.x == 0) *p.due_count = static_cast<int32_t>(count), *p.track_count = static_cast<int32_t>(tcount);
}

/// Track snapshots (prefix cache, extra_buffer): every value-head format region of the pair's slot -> its track slot,
/// with l = l_old + steps + 1, before the working slot is refit (stream order).
template <uint32_t kC>
__global__ void __launch_bounds__(256) gdn_lq_commit_track_kernel(const __grid_constant__ lqr::CommitParams p) {
  using L = gdn_lq::Layout<kC>;
  constexpr uint32_t kVec = L::kL / 16;  // 16-B words before l / lim
  static_assert(L::kL % 16 == 0);
  const uint32_t units = static_cast<uint32_t>(*p.track_count) * p.num_heads;
  for (uint32_t u = blockIdx.x; u < units; u += gridDim.x) {
    const uint32_t q = static_cast<uint32_t>(p.track_list[u / p.num_heads]), h = u % p.num_heads;
    const uint32_t layer = q / p.num_req, i = q % p.num_req;
    const uint8_t* src =
        p.pool + layer * p.layer_stride + p.slot_idx[i] * p.slot_stride + static_cast<int64_t>(h) * lqr::kHeadBytes;
    uint8_t* dst =
        p.pool + layer * p.layer_stride + p.track_idx[i] * p.slot_stride + static_cast<int64_t>(h) * lqr::kHeadBytes;
    for (uint32_t w = threadIdx.x; w < kVec; w += blockDim.x)
      reinterpret_cast<uint4*>(dst)[w] = reinterpret_cast<const uint4*>(src)[w];
    if (threadIdx.x == 0) {
      const int2 ll = *reinterpret_cast<const int2*>(src + L::kL);
      *reinterpret_cast<int2*>(dst + L::kL) = make_int2(p.track_l[q], ll.y);
    }
  }
}

template <uint32_t kC, uint32_t kIters, bool kWarm>
__global__ void __launch_bounds__(256, 2) gdn_lq_commit_refit_kernel(const __grid_constant__ lqr::CommitParams p) {
  using Core = lqr::RefitCore<kC, kIters, kWarm>;
  extern __shared__ __align__(16) uint8_t smem_raw[];
  auto& sm = *reinterpret_cast<typename Core::Smem*>(smem_raw);
  // l / lim of every value-head copy
  const uint32_t pairs = p.num_layers * p.num_req;
  for (uint32_t u = blockIdx.x * blockDim.x + threadIdx.x; u < pairs * p.num_heads; u += gridDim.x * blockDim.x) {
    const uint32_t q = u / p.num_heads, h = u % p.num_heads;
    const int32_t l = p.pair_l[q];
    if (l < 0) continue;
    const uint32_t layer = q / p.num_req, i = q % p.num_req;
    *reinterpret_cast<int2*>(
        p.pool + layer * p.layer_stride + p.slot_idx[i] * p.slot_stride + static_cast<int64_t>(h) * lqr::kHeadBytes +
        p.lofs) = make_int2(l, p.pair_lim[q]);
  }
  const uint32_t units = static_cast<uint32_t>(*p.due_count) * p.num_heads;
  for (uint32_t u = blockIdx.x; u < units; u += gridDim.x) {
    const uint32_t q = static_cast<uint32_t>(p.due_list[u / p.num_heads]), h = u % p.num_heads;
    const uint32_t layer = q / p.num_req, i = q % p.num_req;
    uint8_t* base = p.pool + layer * p.layer_stride + p.slot_idx[i] * p.slot_stride;
    uint8_t* hs = base + static_cast<int64_t>(h) * lqr::kHeadBytes;
    const uint8_t* ks = base + static_cast<int64_t>(h & ~1u) * lqr::kHeadBytes;
    __syncthreads();  // smem reuse across units
    Core core;
    core.run(sm, hs, ks, hs, static_cast<uint32_t>(p.pair_lnew[q]), p.q0, p.floor_c);
  }
}

/// pool [L, S, HV, 128, 128] fp32 (format view); slot_idx [N] int32; n_acc [N] int32 or empty (flush);
/// ws int32 [4 * L * N + 1] workspace.
template <uint32_t kC, uint32_t kIters, bool kWarm>
void gdn_lq_commit(
    tvm::ffi::TensorView pool,
    tvm::ffi::TensorView slot_idx,
    tvm::ffi::TensorView n_acc,
    tvm::ffi::TensorView q0,
    tvm::ffi::TensorView ws,
    tvm::ffi::TensorView track_idx,
    tvm::ffi::TensorView track_steps,
    int64_t P,
    double floor_c) {
  using namespace host;
  using Core = lqr::RefitCore<kC, kIters, kWarm>;
  auto NL = SymbolicSize{"layers"}, NS = SymbolicSize{"slots"}, HV = SymbolicSize{"heads"}, N = SymbolicSize{"reqs"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({NL, NS, HV, 128, 128})  //
      .with_dtype<float>()
      .with_device<kDLCUDA>(device)
      .verify(pool);
  TensorMatcher({N})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device)
      .verify(slot_idx);
  TensorMatcher({128, 4})  //
      .with_dtype<float>()
      .with_device<kDLCUDA>(device)
      .verify(q0);
  const uint32_t nl = static_cast<uint32_t>(NL.unwrap()), n = static_cast<uint32_t>(N.unwrap());
  const uint32_t hv = static_cast<uint32_t>(HV.unwrap());
  TensorMatcher({6 * nl * n + 2})  //
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device)
      .verify(ws);
  const bool has_n = n_acc.numel() > 0;
  if (has_n) {
    TensorMatcher({N})  //
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(n_acc);
  }
  const bool has_track = track_idx.numel() > 0;
  if (has_track) {
    TensorMatcher({N})  //
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(track_idx)
        .verify(track_steps);
  }
  CHECK_HOST(hv % 2 == 0) << "value heads must pair with key heads";
  if (n == 0) return;
  lqr::CommitParams p{};
  p.track_idx = has_track ? static_cast<const int32_t*>(track_idx.data_ptr()) : nullptr;
  p.track_steps = has_track ? static_cast<const int32_t*>(track_steps.data_ptr()) : nullptr;
  p.pool = static_cast<uint8_t*>(pool.data_ptr());
  p.slot_idx = static_cast<const int32_t*>(slot_idx.data_ptr());
  p.n_acc = has_n ? static_cast<const int32_t*>(n_acc.data_ptr()) : nullptr;
  int32_t* w = static_cast<int32_t*>(ws.data_ptr());
  p.pair_l = w;
  p.pair_lim = w + nl * n;
  p.pair_lnew = w + 2 * nl * n;
  p.due_list = w + 3 * nl * n;
  p.due_count = w + 4 * nl * n;
  p.track_list = w + 4 * nl * n + 1;
  p.track_l = w + 5 * nl * n + 1;
  p.track_count = w + 6 * nl * n + 1;
  p.q0 = static_cast<const float*>(q0.data_ptr());
  p.slot_stride = static_cast<int64_t>(hv) * lqr::kHeadBytes;
  p.layer_stride = pool.stride(0) * 4;
  p.num_layers = nl;
  p.num_req = n;
  p.num_heads = hv;
  p.lofs = gdn_lq::Layout<kC>::kL;
  p.P = static_cast<int32_t>(P);
  p.floor_c = static_cast<float>(floor_c);
  const DLDevice dev = device.unwrap();
  LaunchKernel(1, 1024, dev)(gdn_lq_commit_scan_kernel, p);
  if (has_track) LaunchKernel(runtime::get_sm_count(dev.device_id), 256, dev)(gdn_lq_commit_track_kernel<kC>, p);
  constexpr uint32_t kSmem = sizeof(typename Core::Smem);
  runtime::set_max_dynamic_smem_per_device<gdn_lq_commit_refit_kernel<kC, kIters, kWarm>>(dev, kSmem);
  const uint32_t grid = runtime::get_sm_count(dev.device_id) * 2;
  LaunchKernel(grid, Core::kThreads, dev, kSmem)(gdn_lq_commit_refit_kernel<kC, kIters, kWarm>, p);
}

}  // namespace sglang
