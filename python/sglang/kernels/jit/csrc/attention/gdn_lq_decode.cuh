// LeapQuant GDN decode (T = 1) for Qwen3.5: one fused kernel per layer replacing production's
// unpack + conv1d update (D1), packed recurrence (D2) and gated RMSNorm (D3).
//
// Grid (16 key heads, N requests), 256 threads: one CTA = one (request, key head) = 2 value heads; warps 0-3
// own value head 2 hk, warps 4-7 value head 2 hk + 1 (one thread per V row in the row phase).
// Format: gdn_lq_format.cuh (option A, aliases the fp32 temporal slot); l / lim are replicated in every value
// head's region; each CTA reads and writes the copies of its own heads.
// Step semantics (leapquant_ref GDN oracle, decode):
//   conv + SiLU as production (bf16 products, bf16 output), g = -exp(A_log) softplus(a + dt_bias),
//   beta = bf16(sigmoid(b)), k^ = fp16(l2norm(k)), q^ = l2norm(q) K^-1/2, G' = G_l + g,
//   x = e^{G'} {k^, q^}; base read B x = s * (Z (c x)) + U~^T (K~ x) with c x split hi + lo fp16;
//   committed records j < l weighted e^{G' - G_j}; u = beta (v - W0); o = O0 + (q^.k^) u;
//   record l <- (G', k^, fp16(u)); l <- l + 1 (refit outside this kernel for now); gated RMSNorm.
// The Z product runs on mma.sync m16n8k16 (int8 -> f16 exact). Each thread loads 32 contiguous bytes of a Z row
// per fragment row, so the k index inside a k-step is permuted identically in A and B (physical byte
// q4 * 32 + kk * 4 + {0..3} <-> fragment k {2 q4, 2 q4 + 1, 2 q4 + 8, 2 q4 + 9}).

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/ptx/async.cuh>
#include <sgl_kernel/ptx/barrier.cuh>

#include <tvm/ffi/container/tensor.h>

#include "gdn_lq_format.cuh"
#include "gdn_lq_refit.cuh"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace sglang {
namespace {  // experiment-time: keep symbols local per module

namespace gdn_lq_dec {

using gdn_lq::kDim;
inline constexpr uint32_t kC = 20;
inline constexpr uint32_t kP = 16;
inline constexpr uint32_t kHK = 16;
inline constexpr uint32_t kHV = 32;
inline constexpr uint32_t kQKV = 8192;  // conv channels: q 2048 | k 2048 | v 4096; z follows in qkvz
using L = gdn_lq::Layout<kC>;

enum RefitMode { REFIT_NONE = 0, REFIT_WRAP = 1, REFIT_SHELL = 2, REFIT_FUSED = 3, REFIT_LIST = 4 };
using RefitCoreCold1 = lqr::RefitCore<kC, 1, false>;  // oracle default refit (cold1)

struct Params {
  const __nv_bfloat16* __restrict__ qkvz;     // [N, >= 12288]
  const __nv_bfloat16* __restrict__ ba;       // [N, >= 64]: b [32] | a [32]
  const int32_t* __restrict__ idx;            // [N] slot, < 0 = padding
  uint8_t* pool;                              // temporal layer slice, slot stride slot_bytes, head stride 64 KB
  __nv_bfloat16* conv;                        // [slots, 8192, 3]
  const __nv_bfloat16* __restrict__ conv_w;   // [8192, 4]
  const float* __restrict__ A_log;            // [32]
  const __nv_bfloat16* __restrict__ dt_bias;  // [32]
  const __nv_bfloat16* __restrict__ norm_w;   // [128]
  __nv_bfloat16* __restrict__ out;            // [N, 4096]
  float* __restrict__ dbg;                    // [N, 8192 + 4096] conv out (bf16 values) | pre-norm o, or null
  const float* __restrict__ q0;               // [128, 4] refit start block (REFIT_FUSED)
  int32_t* due;                               // REFIT_LIST: [2 + 16 N] = count | done | due units (n * 16 + hk)
  int64_t qkvz_stride, ba_stride, slot_bytes, conv_slot_stride;
  float eps;
};

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

SGL_DEVICE void mma_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// element i of 4 packed 16-bit values (no address taken: keeps the pair in registers)
SGL_DEVICE uint16_t u16_at(uint2 v, int i) {
  const uint32_t w = (i < 2) ? v.x : v.y;
  return static_cast<uint16_t>((i & 1) ? (w >> 16) : (w & 0xFFFFu));
}
SGL_DEVICE float half_at(uint2 v, int i) {
  return __half2float(__ushort_as_half(u16_at(v, i)));
}

SGL_DEVICE float bf(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
SGL_DEVICE float bf16_round(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}
SGL_DEVICE float ex2_approx(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// Production conv (triton_gdn_fused_proj.py): acc += bf16(state * w) for the 3 state taps, + bf16(x * w3),
// silu = acc / (1 + exp(-acc)).
SGL_DEVICE float conv_silu(const __nv_bfloat16 (&s)[3], __nv_bfloat16 x, uint2 wv) {
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < 3; ++i)
    acc += bf(__hmul(s[i], __ushort_as_bfloat16(u16_at(wv, i))));
  acc += bf(__hmul(x, __ushort_as_bfloat16(u16_at(wv, 3))));
  const float e = ex2_approx(-acc * 1.4426950408889634f);
  return __fdiv_rn(acc, 1.f + e);
}

SGL_DEVICE void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
               "l"(src)
               : "memory");
}
SGL_DEVICE void cp_wait_all() {
  asm volatile("cp.async.wait_all;" ::: "memory");
}
SGL_DEVICE void ldsm_x4(uint32_t (&r)[4], const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
SGL_DEVICE void ldsm_x4_trans(uint32_t (&r)[4], const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

// fp16 operands of the dot and record products are scaled by powers of two into the normal range (|q^| <= 1,
// |w|, |t| <= ~1): exact, undone on the fp32 accumulators.
inline constexpr float kDotScale = 256.f;
inline constexpr float kRecScale = 16384.f;

// position of B element (k, n) of one m16n8k16 k-step inside [16 lanes][4 halves] (n < 4)
SGL_DEVICE uint32_t bfrag_idx(uint32_t k, uint32_t n) {
  const uint32_t w = k % 16;
  const uint32_t q4 = (w % 8) / 2;
  const uint32_t half = (w >= 8 ? 2 : 0) + (w % 2);
  return (n * 4 + q4) * 4 + half;
}

// Dynamic shared memory: the slot-dependent state staged by cp.async (no registers held while in flight).
struct Smem {
  static constexpr uint32_t kZPitch = kDim + 16;  // 144 B rows: conflict-free 16 B fragment reads
  static constexpr uint32_t kHPitch = kDim + 8;   // 272 B fp16 rows: conflict-free ldmatrix
  uint8_t z[2][kDim][kZPitch];
  __half ur[2][32][kHPitch];  // A^T of the record product: rows j 0..19 u_j, 20..23 U~_r, 24..31 zero
  __half ad[32][kHPitch];     // dot A: rows 0..19 k^_j, 20..23 K~ (head 0), 24..27 K~ (head 1), 28 k^, 29..31 0
  float sc[2][2][kDim];       // s | c per head
  float x[512];               // conv out: q | k | v (head 0) | v (head 1)
  float zz[256];              // output gate z per (head, row)
  float nw[kDim];
  __align__(16) __half Bz[2][8][16][4];  // Z product B (permuted k): k_hi, q_hi, k_lo, q_lo
  __align__(16) __half Bd[8][16][4];     // dot B: k^, q^_hi, q^_lo
  __align__(16) __half Br[2][2][16][4];  // record B: wk_hi, wq_hi, wk_lo, wq_lo
  float g[2][32];                        // record log-decays G_j per head
  float gate[2][4];                      // G', e^G', beta per head
  float qk;
  float red[8];
};

template <bool kDebug, int kRefit>
__global__ void __launch_bounds__(256) decode_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
  const uint32_t hk = blockIdx.x;
  const uint32_t n = blockIdx.y;
  const uint32_t t = threadIdx.x;
  const uint32_t warp = t / 32;
  const uint32_t lane = t % 32;
  const uint32_t h_row = t / 128;  // value head of thread t's column / row role (= warp / 4)
  const uint32_t v_row = t % 128;

  const int32_t slot = p.idx[n];
  // ---- inputs (slot independent) ----
  const __nv_bfloat16* xrow = p.qkvz + n * p.qkvz_stride;
  const uint32_t ch0 = (t < 128) ? hk * kDim + t : 2048 + hk * kDim + (t - 128);  // q or k
  const uint32_t ch1 = 4096 + (2 * hk + h_row) * kDim + v_row;                    // v of head h_row
  const __nv_bfloat16 x0 = xrow[ch0];
  const __nv_bfloat16 x1 = xrow[ch1];
  const __nv_bfloat16 zz = xrow[kQKV + (2 * hk + h_row) * kDim + v_row];
  const uint2 w0 = *reinterpret_cast<const uint2*>(p.conv_w + ch0 * 4);
  const uint2 w1 = *reinterpret_cast<const uint2*>(p.conv_w + ch1 * 4);
  if (slot < 0) {
    p.out[n * (kHV * kDim) + (2 * hk + h_row) * kDim + v_row] = __float2bfloat16_rn(0.f);
    return;
  }

  // ---- state: issue every copy first (cp.async, 16 B), then the conv while it lands ----
  uint8_t* const slot_base = p.pool + slot * p.slot_bytes;
  uint8_t* const reg0 = slot_base + (2 * hk) * gdn_lq::kHeadBytes;  // even head: k^ records, l, lim
  {
#pragma unroll
    for (int i = 0; i < 8; ++i) {  // Z: 2 heads x 128 rows x 8 chunks
      const uint32_t c = t + 256 * i, h = c / 1024, r = (c / 8) % 128, k = c % 8;
      cp16(&sm.z[h][r][k * 16], reg0 + h * gdn_lq::kHeadBytes + L::kZ + r * kDim + k * 16);
    }
    for (uint32_t c = t; c < 640; c += 256) {  // u records: 2 heads x 20 rows x 16 chunks
      const uint32_t h = c / 320, j = (c / 16) % kC, k = c % 16;
      cp16(&sm.ur[h][j][k * 8], reg0 + h * gdn_lq::kHeadBytes + L::kU + (j * kDim + k * 8) * 2);
    }
    for (uint32_t c = t; c < 320; c += 256)
      cp16(&sm.ad[c / 16][(c % 16) * 8], reg0 + L::kKh + c * 16);
    {  // per head: U~ 4 rows | K~ 4 rows, 16 chunks each
      const uint32_t h = t / 128, k = t % 128, r = (k % 64) / 16, cc = k % 16;
      if (k < 64)
        cp16(&sm.ur[h][20 + r][cc * 8], reg0 + h * gdn_lq::kHeadBytes + L::kUt + (r * kDim + cc * 8) * 2);
      else
        cp16(&sm.ad[20 + 4 * h + r][cc * 8], reg0 + h * gdn_lq::kHeadBytes + L::kKt + (r * kDim + cc * 8) * 2);
    }
    if (t < 128) {  // per head: s (32 chunks) | c (32 chunks), contiguous in the format
      const uint32_t h = t / 64, k = t % 64;
      cp16(&sm.sc[h][0][0] + k * 4, reg0 + h * gdn_lq::kHeadBytes + L::kS + k * 16);
    } else if (t < 138) {  // per head: G 5 chunks
      const uint32_t h = (t - 128) / 5, k = (t - 128) % 5;
      cp16(&sm.g[h][k * 4], reg0 + h * gdn_lq::kHeadBytes + L::kG + k * 16);
    }
  }
  const int32_t l = *reinterpret_cast<const int32_t*>(reg0 + L::kL);
  [[maybe_unused]] int32_t lim = *reinterpret_cast<const int32_t*>(reg0 + L::kLim);
  lim = lim == 0 ? int32_t(kP) : lim;
  __nv_bfloat16 cs0[3], cs1[3];
  __nv_bfloat16* const conv_slot = p.conv + slot * p.conv_slot_stride;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    cs0[i] = conv_slot[ch0 * 3 + i];
    cs1[i] = conv_slot[ch1 * 3 + i];
  }
  // gates: one thread per value head
  if (t < 2) {
    const uint32_t h = t;
    const float a = bf(p.ba[n * p.ba_stride + kHV + 2 * hk + h]);
    const float b = bf(p.ba[n * p.ba_stride + 2 * hk + h]);
    const float x = a + bf(p.dt_bias[2 * hk + h]);
    const float sp = x <= 20.f ? log1pf(expf(x)) : x;
    const float g = -expf(p.A_log[2 * hk + h]) * sp;
    const float Gl = l > 0 ? *reinterpret_cast<const float*>(reg0 + h * gdn_lq::kHeadBytes + L::kG + (l - 1) * 4) : 0.f;
    const float Gn = Gl + g;
    sm.gate[h][0] = Gn;
    sm.gate[h][1] = expf(Gn);
    sm.gate[h][2] = bf16_round(1.f / (1.f + expf(-b)));
  }
  // zero fill: record-A rows 24..31 (both heads), dot-A rows 29..31, record B
  {
    const uint4 z4 = make_uint4(0, 0, 0, 0);
    for (uint32_t c = t; c < 2 * 8 * 17; c += 256) {
      const uint32_t h = c / 136, r = (c / 17) % 8, k = c % 17;
      *reinterpret_cast<uint4*>(&sm.ur[h][24 + r][k * 8]) = z4;
    }
    if (t < 3 * 17) *reinterpret_cast<uint4*>(&sm.ad[29 + t / 17][(t % 17) * 8]) = z4;
    if (t < 32) *reinterpret_cast<uint4*>(&sm.Br[0][0][0][0] + t * 8) = z4;
  }
  sm.zz[t] = bf(zz);
  if (t < kDim) sm.nw[t] = bf(p.norm_w[t]);

  // ---- conv + SiLU (bf16 output, production numerics) ----
  {
    const float o0 = bf16_round(conv_silu(cs0, x0, w0));
    const float o1 = bf16_round(conv_silu(cs1, x1, w1));
    sm.x[t] = o0;
    sm.x[256 + t] = o1;
    conv_slot[ch0 * 3 + 0] = cs0[1];
    conv_slot[ch0 * 3 + 1] = cs0[2];
    conv_slot[ch0 * 3 + 2] = x0;
    conv_slot[ch1 * 3 + 0] = cs1[1];
    conv_slot[ch1 * 3 + 1] = cs1[2];
    conv_slot[ch1 * 3 + 2] = x1;
    if constexpr (kDebug) {
      float* d = p.dbg + n * (kQKV + kHV * kDim);
      d[ch0] = o0;
      d[ch1] = o1;
    }
  }
  cp_wait_all();
  __syncthreads();
  if constexpr (kRefit == REFIT_SHELL) {
    // shell: same loads and stores, no math (every staged buffer feeds a stored value)
    const uint32_t h = h_row, v = v_row;
    float acc = sm.x[t] + sm.x[256 + t] + sm.sc[h][0][v] + sm.sc[h][1][v] + sm.g[h][v % 32] + sm.zz[t] + sm.nw[v];
    acc += __half2float(sm.ur[h][v % 24][v]) + __half2float(sm.ad[v % 28][v]);
#pragma unroll
    for (int i = 0; i < 8; ++i)
      acc += float(reinterpret_cast<const int8_t*>(&sm.z[h][v][0])[i * 16]) + sm.gate[h][0];
    *reinterpret_cast<__half*>(reg0 + h * gdn_lq::kHeadBytes + L::kU + (l * kDim + v) * 2) = __float2half_rn(acc);
    if (h == 0) *reinterpret_cast<__half*>(reg0 + L::kKh + (l * kDim + v) * 2) = __float2half_rn(acc);
    if (v == 0) *reinterpret_cast<float*>(reg0 + h * gdn_lq::kHeadBytes + L::kG + l * 4) = acc;
    p.out[n * (kHV * kDim) + (2 * hk + h) * kDim + v] = __float2bfloat16_rn(acc);
    if (t < 2) *reinterpret_cast<int32_t*>(reg0 + t * gdn_lq::kHeadBytes + L::kL) = (l + 1) >= lim ? 0 : l + 1;
    return;
  }

  // ---- prep: l2 norms (every warp redundantly), B operands ----
  float rq, rk;
  {
    const float4 qv = *reinterpret_cast<const float4*>(&sm.x[lane * 4]);
    const float4 kv = *reinterpret_cast<const float4*>(&sm.x[128 + lane * 4]);
    const float sq = device::warp::reduce_sum(qv.x * qv.x + qv.y * qv.y + qv.z * qv.z + qv.w * qv.w);
    const float sk = device::warp::reduce_sum(kv.x * kv.x + kv.y * kv.y + kv.z * kv.z + kv.w * kv.w);
    rq = (1.f / sqrtf(sq + 1e-6f)) * 0.08838834764831845f;
    rk = 1.f / sqrtf(sk + 1e-6f);
  }
  const float Gn_row = sm.gate[h_row][0];
  const float eG_row = sm.gate[h_row][1];
  const float beta_row = sm.gate[h_row][2];
  {
    const uint32_t i = v_row;
    const float c_col = sm.sc[h_row][1][i];
    const __half khh = __float2half_rn(sm.x[128 + i] * rk);
    const float qh = sm.x[i] * rq;
    const float cx_k = c_col * (eG_row * __half2float(khh));
    const float cx_q = c_col * (eG_row * qh);
    const __half kh_hi = __float2half_rn(cx_k), qh_hi = __float2half_rn(cx_q);
    const __half kh_lo = __float2half_rn(cx_k - __half2float(kh_hi));
    const __half qh_lo = __float2half_rn(cx_q - __half2float(qh_hi));
    const uint32_t qq = i / 32, kk = (i % 32) / 4, sub = i % 4;  // permuted k of the Z product
    sm.Bz[h_row][kk][0 * 4 + qq][sub] = kh_hi;
    sm.Bz[h_row][kk][1 * 4 + qq][sub] = qh_hi;
    sm.Bz[h_row][kk][2 * 4 + qq][sub] = kh_lo;
    sm.Bz[h_row][kk][3 * 4 + qq][sub] = qh_lo;
    if (h_row == 0) {
      const float qs = qh * kDotScale;
      const __half q_hi = __float2half_rn(qs);
      const __half q_lo = __float2half_rn(qs - __half2float(q_hi));
      __half* bd = &sm.Bd[i / 16][0][0];
      bd[bfrag_idx(i, 0)] = khh;
      bd[bfrag_idx(i, 1)] = q_hi;
      bd[bfrag_idx(i, 2)] = q_lo;
      bd[bfrag_idx(i, 3)] = __float2half_rn(0.f);
      sm.ad[28][i] = khh;
      *reinterpret_cast<__half*>(reg0 + L::kKh + (l * kDim + i) * 2) = khh;
    }
  }
  __syncthreads();

  // ---- dots on mma.sync (warps 0, 1): rows j of [k^_j | K~ | k^] . [k^, q^_hi, q^_lo] -> record B ----
  if (warp < 2) {
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    const uint32_t mrow = warp * 16 + (lane % 8) + ((lane / 8) % 2) * 8;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t a[4];
      ldsm_x4(a, &sm.ad[mrow][ks * 16 + (lane / 16) * 8]);
      const uint2 b = (lane < 16) ? *reinterpret_cast<const uint2*>(&sm.Bd[ks][lane][0]) : make_uint2(0, 0);
      mma_16816(d, a, b.x, b.y);
    }
    // q4 = 0 lanes: (dk, dq_hi) of rows j, j + 8; q4 = 1 lanes: (dq_lo, 0)
    const float lo0 = __shfl_xor_sync(0xffffffffu, d[0], 1);
    const float lo1 = __shfl_xor_sync(0xffffffffu, d[2], 1);
    if (lane % 4 == 0) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const uint32_t j = warp * 16 + lane / 4 + 8 * e;
        const float dk = e ? d[2] : d[0];
        const float dq = ((e ? d[3] : d[1]) + (e ? lo1 : lo0)) * (1.f / kDotScale);
        if (j < kC) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            float w = 0.f;
            if (int32_t(j) < l) w = expf(sm.gate[h][0] - sm.g[h][j]);
            const float wk = dk * w * kRecScale, wq = dq * w * kRecScale;
            __half* br = &sm.Br[h][j / 16][0][0];
            const __half wk_hi = __float2half_rn(wk), wq_hi = __float2half_rn(wq);
            br[bfrag_idx(j, 0)] = wk_hi;
            br[bfrag_idx(j, 1)] = wq_hi;
            br[bfrag_idx(j, 2)] = __float2half_rn(wk - __half2float(wk_hi));
            br[bfrag_idx(j, 3)] = __float2half_rn(wq - __half2float(wq_hi));
          }
        } else if (j < 28) {
          const uint32_t h = (j - 20) / 4, k = 20 + (j - 20) % 4;
          const float tk = sm.gate[h][1] * dk * kRecScale, tq = sm.gate[h][1] * dq * kRecScale;
          __half* br = &sm.Br[h][1][0][0];
          const __half tk_hi = __float2half_rn(tk), tq_hi = __float2half_rn(tq);
          br[bfrag_idx(k, 0)] = tk_hi;
          br[bfrag_idx(k, 1)] = tq_hi;
          br[bfrag_idx(k, 2)] = __float2half_rn(tk - __half2float(tk_hi));
          br[bfrag_idx(k, 3)] = __float2half_rn(tq - __half2float(tq_hi));
        } else if (j == 28) {
          sm.qk = dq;
        }
      }
    }
  }

  // ---- base read: Z (int8 -> f16) x [k_hi q_hi k_lo q_lo] on mma.sync ----
  const uint32_t mh = warp / 4;
  const uint32_t rb = (warp % 4) * 32;
  const uint32_t r0 = lane / 4, q4 = lane % 4;
  float dz[2][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    const uint32_t R = rb + mt * 16 + r0;
    uint4 zr[2][2];
    zr[0][0] = *reinterpret_cast<const uint4*>(&sm.z[mh][R][q4 * 32]);
    zr[0][1] = *reinterpret_cast<const uint4*>(&sm.z[mh][R][q4 * 32 + 16]);
    zr[1][0] = *reinterpret_cast<const uint4*>(&sm.z[mh][R + 8][q4 * 32]);
    zr[1][1] = *reinterpret_cast<const uint4*>(&sm.z[mh][R + 8][q4 * 32 + 16]);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      dz[mt][i] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      const uint4& h0 = zr[0][kk / 4];
      const uint4& h1 = zr[1][kk / 4];
      const uint32_t wr = (kk % 4 == 0) ? h0.x : (kk % 4 == 1) ? h0.y : (kk % 4 == 2) ? h0.z : h0.w;
      const uint32_t wr8 = (kk % 4 == 0) ? h1.x : (kk % 4 == 1) ? h1.y : (kk % 4 == 2) ? h1.z : h1.w;
      uint32_t a[4];
      i8x4_to_f16x2(wr, a[0], a[2]);
      i8x4_to_f16x2(wr8, a[1], a[3]);
      uint2 b = make_uint2(0, 0);
      if (r0 < 4) b = *reinterpret_cast<const uint2*>(&sm.Bz[mh][kk][lane][0]);
      mma_16816(dz[mt], a, b.x, b.y);
    }
  }
  __syncthreads();  // record B (dot warps) ready

  // ---- record + compensator product: [u_j | U~_r]^T (ldmatrix.trans) x [wk, wq hi / lo] ----
  float dr[2][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
      dr[mt][i] = 0.f;
    const uint32_t m0 = rb + mt * 16;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      uint32_t a[4];
      ldsm_x4_trans(a, &sm.ur[mh][ks * 16 + (lane % 8) + (lane / 16) * 8][m0 + ((lane / 8) % 2) * 8]);
      const uint2 b = (lane < 16) ? *reinterpret_cast<const uint2*>(&sm.Br[mh][ks][lane][0]) : make_uint2(0, 0);
      mma_16816(dr[mt], a, b.x, b.y);
    }
  }

  // ---- finalize: fold hi + lo of the fragments, then one row per lane ----
  const float qk = sm.qk;
  float wv[2][2], ov[2][2];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const uint32_t R = rb + mt * 16 + r0 + 8 * e;
      const float s_v = sm.sc[mh][0][R];
      const float pw = s_v * dz[mt][2 * e] + dr[mt][2 * e] * (1.f / kRecScale);  // q4 0: hi, q4 1: lo parts
      const float po = s_v * dz[mt][2 * e + 1] + dr[mt][2 * e + 1] * (1.f / kRecScale);
      wv[mt][e] = pw + __shfl_xor_sync(0xffffffffu, pw, 1);
      ov[mt][e] = po + __shfl_xor_sync(0xffffffffu, po, 1);
    }
  }
  // lane L takes row rb + L: (m-tile, half, r0) = (L / 16, (L / 8) % 2, L % 8), held by lane 4 (L % 8)
  float wsum = 0.f, osum = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float a = __shfl_sync(0xffffffffu, wv[i / 2][i % 2], 4 * (lane % 8));
    const float b = __shfl_sync(0xffffffffu, ov[i / 2][i % 2], 4 * (lane % 8));
    if (lane / 8 == uint32_t(i)) wsum = a, osum = b;
  }
  const uint32_t R = rb + lane;
  const float U = beta_row * (sm.x[256 + mh * 128 + R] - wsum);
  const float o = osum + qk * U;
  *reinterpret_cast<__half*>(reg0 + mh * gdn_lq::kHeadBytes + L::kU + (l * kDim + R) * 2) = __float2half_rn(U);
  if constexpr (kDebug) p.dbg[n * (kQKV + kHV * kDim) + kQKV + (2 * hk + mh) * kDim + R] = o;
  if (t % 128 == 0) *reinterpret_cast<float*>(reg0 + mh * gdn_lq::kHeadBytes + L::kG + l * 4) = Gn_row;
  const float ss = device::warp::reduce_sum(o * o);
  if (lane == 0) sm.red[warp] = ss;
  device::ptx::bar_sync(1 + mh, 128);
  const float tot = sm.red[4 * mh] + sm.red[4 * mh + 1] + sm.red[4 * mh + 2] + sm.red[4 * mh + 3];
  const float rn = rsqrtf(tot * (1.f / kDim) + p.eps);
  const float zf = sm.zz[mh * 128 + R];
  const float y = o * rn * sm.nw[R] * (zf / (1.f + expf(-zf)));
  p.out[n * (kHV * kDim) + (2 * hk + mh) * kDim + R] = __float2bfloat16_rn(y);
  if constexpr (kRefit == REFIT_LIST) {
    if (t == 0 && l + 1 >= lim) p.due[2 + atomicAdd(&p.due[0], 1)] = static_cast<int32_t>(n * kHK + hk);
  }
  if constexpr (kRefit == REFIT_FUSED) {
    if (l + 1 >= lim) {  // the window is full: refit both value heads in place (CTA-uniform branch)
      __syncthreads();   // this step's records (global, same CTA) are visible to the refit's loads
      auto& rs = *reinterpret_cast<typename RefitCoreCold1::Smem*>(smem_raw);
#pragma unroll 1
      for (uint32_t h = 0; h < 2; ++h) {
        RefitCoreCold1 core;
        uint8_t* hs = reg0 + h * gdn_lq::kHeadBytes;
        core.run(rs, hs, reg0, hs, static_cast<uint32_t>(l + 1), p.q0, 1e-6f);
        __syncthreads();
      }
      if (t < 2) {
        *reinterpret_cast<int32_t*>(reg0 + t * gdn_lq::kHeadBytes + L::kL) = 0;
        *reinterpret_cast<int32_t*>(reg0 + t * gdn_lq::kHeadBytes + L::kLim) = int32_t(kP);
      }
      return;
    }
  }
  if (t < 2) {  // l is replicated in every value head region: this CTA owns heads 2 hk, 2 hk + 1
    int32_t ln = l + 1;
    if constexpr (kRefit == REFIT_WRAP) ln = ln >= lim ? 0 : ln;
    *reinterpret_cast<int32_t*>(reg0 + t * gdn_lq::kHeadBytes + L::kL) = ln;
  }
}

// ================================================================== persistent, warp-specialized variant
// One CTA per SM: a producer warp streams each unit (request, key head) into a ring of kStages stages with
// cp.async.bulk (mbarrier complete_tx); kGroups consumer groups of 8 warps each process one unit at a time
// (unit k of this CTA -> stage k % kStages, group k % kGroups). A group copies what it needs out of the stage
// (Z into registers, records into padded group buffers), releases the stage, then computes as the grid kernel.

struct Stage {                       // byte offsets inside one stage (each block 16 B aligned, bulk-copy destinations)
  static constexpr uint32_t kZ = 0;  // Z head 0 | head 1 (2 x 16 KB)
  static constexpr uint32_t kU = kZ + 2 * 16384;  // per head: u [20][128] fp16 | U~ [4][128]
  static constexpr uint32_t kUBytes = (kC + 4) * kDim * 2;
  static constexpr uint32_t kAD = kU + 2 * kUBytes;           // k^ records [20][128] | K~ head 0 [4] | K~ head 1 [4]
  static constexpr uint32_t kSC = kAD + (kC + 8) * kDim * 2;  // per head s | c (1 KB)
  static constexpr uint32_t kG = kSC + 2 * 1024;              // per head G [20] | l | lim | pad (96 B)
  static constexpr uint32_t kQ = kG + 2 * 96;                 // q 256 | k 256 | v 512 | z 512
  static constexpr uint32_t kCS = kQ + 1536;                  // conv state: q 768 | k 768 | v 1536
  static constexpr uint32_t kCW = kCS + 3072;                 // conv weights: q 1024 | k 1024 | v 2048
  static constexpr uint32_t kBytes = kCW + 4096;
  static_assert(kBytes % 16 == 0 && kU % 16 == 0 && kAD % 16 == 0 && kSC % 16 == 0 && kG % 16 == 0);
};

struct GroupSmem {
  static constexpr uint32_t kHPitch = kDim + 8;
  __half ur[2][kC + 4][kHPitch];  // rows >= 24 of the k-steps read the shared zero row
  __half ad[kC + 9][kHPitch];     // rows >= 29 read the shared zero row
  float sc[2][2][kDim];
  float g[2][32];
  float x[512];
  float zz[256];
  __align__(16) __half Bz[2][8][16][4];
  __align__(16) __half Bd[8][16][4];
  __align__(16) __half Br[2][2][16][4];
  float gate[2][4];
  float qk;
  float red[8];
};

template <uint32_t kGroups, uint32_t kStages>
struct PersistentSmem {
  uint8_t stage[kStages][Stage::kBytes];
  GroupSmem grp[kGroups];
  __align__(16) __half zero_row[GroupSmem::kHPitch];
  float nw[kDim];
  uint64_t full[kStages];
  uint64_t empty[kStages];
};

template <uint32_t kGroups, uint32_t kStages, bool kDebug, int kRefit>
__global__ void __launch_bounds__(kGroups * 256 + 32, 1)
    decode_persistent_kernel(const __grid_constant__ Params p, uint32_t num_units) {
  // a stage must always be consumed by the same group (in unit order), else a group can wait on an aliased
  // mbarrier phase: unit k -> stage k % kStages, group k % kGroups, so kGroups must divide kStages
  static_assert(kStages % kGroups == 0);
  using SM = PersistentSmem<kGroups, kStages>;
  extern __shared__ __align__(128) uint8_t smem_raw[];
  SM& S = *reinterpret_cast<SM*>(smem_raw);
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_all = tid / 32;

  // one-time: barriers, zero rows of the group buffers (never written by a unit), norm weight
  if (tid == 0) {
    for (uint32_t i = 0; i < kStages; ++i) {
      device::ptx::mbar_init(&S.full[i], 1);
      device::ptx::mbar_init(&S.empty[i], 1);
    }
    device::ptx::fence_barrier_init();
  }
  if (tid < 17) *reinterpret_cast<uint4*>(&S.zero_row[tid * 8]) = make_uint4(0, 0, 0, 0);
  if (tid < kDim) S.nw[tid] = bf(p.norm_w[tid]);
  __syncthreads();

  if (warp_all == kGroups * 8) {
    // ---------------- producer ----------------
    // slot indices of the next 32 units are loaded by the 32 lanes ahead of time (off the issue path)
    const uint32_t lane_p = tid % 32;
    int32_t slot_batch = 0;
    for (uint32_t k = 0;; ++k) {
      const uint32_t unit = blockIdx.x + k * gridDim.x;
      if (unit >= num_units) break;
      if (k % 32 == 0) {
        const uint32_t u_l = blockIdx.x + (k + lane_p) * gridDim.x;
        slot_batch = u_l < num_units ? p.idx[u_l / kHK] : -1;
      }
      const int32_t slot = __shfl_sync(0xffffffffu, slot_batch, k % 32);
      if (lane_p == 0) {
        const uint32_t s = k % kStages;
        if (k >= kStages) device::ptx::mbar_wait_parity(&S.empty[s], ((k / kStages) - 1) & 1);
        const uint32_t n = unit / kHK, hk = unit % kHK;
        if (slot < 0) {
          device::ptx::mbar_arrive(&S.full[s]);
        } else {
          uint8_t* st = S.stage[s];
          uint64_t* bar = &S.full[s];
          device::ptx::mbar_arrive_expect_tx(bar, Stage::kBytes);
          const uint8_t* reg0 = p.pool + slot * p.slot_bytes + (2 * hk) * gdn_lq::kHeadBytes;
          const uint8_t* reg1 = reg0 + gdn_lq::kHeadBytes;
          using device::ptx::cp_async_bulk_g2s;
          cp_async_bulk_g2s(st + Stage::kZ, reg0 + L::kZ, 16384, bar);
          cp_async_bulk_g2s(st + Stage::kZ + 16384, reg1 + L::kZ, 16384, bar);
          cp_async_bulk_g2s(st + Stage::kU, reg0 + L::kU, kC * kDim * 2, bar);
          cp_async_bulk_g2s(st + Stage::kU + kC * kDim * 2, reg0 + L::kUt, 4 * kDim * 2, bar);
          cp_async_bulk_g2s(st + Stage::kU + Stage::kUBytes, reg1 + L::kU, kC * kDim * 2, bar);
          cp_async_bulk_g2s(st + Stage::kU + Stage::kUBytes + kC * kDim * 2, reg1 + L::kUt, 4 * kDim * 2, bar);
          cp_async_bulk_g2s(st + Stage::kAD, reg0 + L::kKh, kC * kDim * 2, bar);
          cp_async_bulk_g2s(st + Stage::kAD + kC * kDim * 2, reg0 + L::kKt, 4 * kDim * 2, bar);
          cp_async_bulk_g2s(st + Stage::kAD + (kC + 4) * kDim * 2, reg1 + L::kKt, 4 * kDim * 2, bar);
          cp_async_bulk_g2s(st + Stage::kSC, reg0 + L::kS, 1024, bar);
          cp_async_bulk_g2s(st + Stage::kSC + 1024, reg1 + L::kS, 1024, bar);
          cp_async_bulk_g2s(st + Stage::kG, reg0 + L::kG, 96, bar);
          cp_async_bulk_g2s(st + Stage::kG + 96, reg1 + L::kG, 96, bar);
          const __nv_bfloat16* xrow = p.qkvz + n * p.qkvz_stride;
          cp_async_bulk_g2s(st + Stage::kQ, xrow + hk * kDim, 256, bar);
          cp_async_bulk_g2s(st + Stage::kQ + 256, xrow + 2048 + hk * kDim, 256, bar);
          cp_async_bulk_g2s(st + Stage::kQ + 512, xrow + 4096 + 2 * hk * kDim, 512, bar);
          cp_async_bulk_g2s(st + Stage::kQ + 1024, xrow + kQKV + 2 * hk * kDim, 512, bar);
          const __nv_bfloat16* cs = p.conv + slot * p.conv_slot_stride;
          cp_async_bulk_g2s(st + Stage::kCS, cs + hk * kDim * 3, 768, bar);
          cp_async_bulk_g2s(st + Stage::kCS + 768, cs + (2048 + hk * kDim) * 3, 768, bar);
          cp_async_bulk_g2s(st + Stage::kCS + 1536, cs + (4096 + 2 * hk * kDim) * 3, 1536, bar);
          cp_async_bulk_g2s(st + Stage::kCW, p.conv_w + hk * kDim * 4, 1024, bar);
          cp_async_bulk_g2s(st + Stage::kCW + 1024, p.conv_w + (2048 + hk * kDim) * 4, 1024, bar);
          cp_async_bulk_g2s(st + Stage::kCW + 2048, p.conv_w + (4096 + 2 * hk * kDim) * 4, 2048, bar);
        }
      }
      __syncwarp();
    }
    return;
  }

  // ---------------- consumer group ----------------
  const uint32_t g = warp_all / 8;
  const uint32_t t = tid % 256;
  const uint32_t warp = t / 32;
  const uint32_t lane = t % 32;
  const uint32_t h_row = t / 128;
  const uint32_t v_row = t % 128;
  const uint32_t bar_grp = 1 + 3 * g;
  GroupSmem& sm = S.grp[g];
  const uint32_t mh = warp / 4;
  const uint32_t rb = (warp % 4) * 32;
  const uint32_t r0 = lane / 4, q4 = lane % 4;

  for (uint32_t k = g;; k += kGroups) {
    const uint32_t unit = blockIdx.x + k * gridDim.x;
    if (unit >= num_units) break;
    const uint32_t s = k % kStages;
    const uint32_t n = unit / kHK, hk = unit % kHK;
    const int32_t slot = p.idx[n];
    float a_g = 0.f, b_g = 0.f, dtb_g = 0.f, alog_g = 0.f;
    if (t < 2) {
      b_g = bf(p.ba[n * p.ba_stride + 2 * hk + t]);
      a_g = bf(p.ba[n * p.ba_stride + kHV + 2 * hk + t]);
      dtb_g = bf(p.dt_bias[2 * hk + t]);
      alog_g = p.A_log[2 * hk + t];
    }
    device::ptx::mbar_wait_parity(&S.full[s], (k / kStages) & 1);
    if (slot < 0) {
      p.out[n * (kHV * kDim) + (2 * hk + h_row) * kDim + v_row] = __float2bfloat16_rn(0.f);
      device::ptx::bar_sync(bar_grp, 256);
      if (t == 0) device::ptx::mbar_arrive(&S.empty[s]);
      continue;
    }
    const uint8_t* st = S.stage[s];
    uint8_t* const reg0 = p.pool + slot * p.slot_bytes + (2 * hk) * gdn_lq::kHeadBytes;

    // ---- copy out of the stage ----
    uint4 zr[2][2][2];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int rs = 0; rs < 2; ++rs) {
        const uint8_t* src = st + Stage::kZ + mh * 16384 + (rb + mt * 16 + rs * 8 + r0) * kDim + q4 * 32;
        zr[mt][rs][0] = *reinterpret_cast<const uint4*>(src);
        zr[mt][rs][1] = *reinterpret_cast<const uint4*>(src + 16);
      }
    static_assert(2 * (kC + 4) * 16 == 3 * 256 && (kC + 8) * 16 <= 2 * 256);
    {  // u + U~ rows of both heads (48 rows of 16 chunks) and k^ + K~ rows (28 rows) -> padded group buffers
      uint4 v[5];
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        const uint32_t c = t + 256 * i, row = c / 16, kk = c % 16;  // row over [head 0 rows | head 1 rows]
        v[i] = *reinterpret_cast<const uint4*>(st + Stage::kU + row * (kDim * 2) + kk * 16);
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint32_t c = t + 256 * i;
        if (c < (kC + 8) * 16) v[3 + i] = *reinterpret_cast<const uint4*>(st + Stage::kAD + c * 16);
      }
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        const uint32_t c = t + 256 * i, row = c / 16, kk = c % 16;
        *reinterpret_cast<uint4*>(&sm.ur[row / (kC + 4)][row % (kC + 4)][kk * 8]) = v[i];
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint32_t c = t + 256 * i;
        if (c < (kC + 8) * 16) *reinterpret_cast<uint4*>(&sm.ad[c / 16][(c % 16) * 8]) = v[3 + i];
      }
    }
    if (t < 128)
      *reinterpret_cast<float4*>(&sm.sc[0][0][0] + t * 4) = *reinterpret_cast<const float4*>(st + Stage::kSC + t * 16);
    if (t < 2 * kC)
      sm.g[t / kC][t % kC] = *reinterpret_cast<const float*>(st + Stage::kG + (t / kC) * 96 + (t % kC) * 4);
    const int32_t l = *reinterpret_cast<const int32_t*>(st + Stage::kG + kC * 4);
    int32_t lim = *reinterpret_cast<const int32_t*>(st + Stage::kG + kC * 4 + 4);
    lim = lim == 0 ? int32_t(kP) : lim;
    sm.zz[t] = bf(reinterpret_cast<const __nv_bfloat16*>(st + Stage::kQ + 1024)[t]);
    // conv: channel role as the grid kernel (thread t: q / k channel t, v channel of (head h_row, row v_row))
    {
      const __nv_bfloat16* xs = reinterpret_cast<const __nv_bfloat16*>(st + Stage::kQ);
      const __nv_bfloat16* css = reinterpret_cast<const __nv_bfloat16*>(st + Stage::kCS);
      const uint2* cws = reinterpret_cast<const uint2*>(st + Stage::kCW);
      // stage order q | k | v: element index t for q/k, 256 + t for v
      const __nv_bfloat16 x0 = xs[t], x1 = xs[256 + t];
      __nv_bfloat16 cs0[3], cs1[3];
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        cs0[i] = css[t * 3 + i];
        cs1[i] = css[(256 + t) * 3 + i];
      }
      const uint2 w0 = cws[t], w1 = cws[256 + t];
      const float o0 = bf16_round(conv_silu(cs0, x0, w0));
      const float o1 = bf16_round(conv_silu(cs1, x1, w1));
      sm.x[t] = o0;
      sm.x[256 + t] = o1;
      const uint32_t ch0 = (t < 128) ? hk * kDim + t : 2048 + hk * kDim + (t - 128);
      const uint32_t ch1 = 4096 + (2 * hk + h_row) * kDim + v_row;
      __nv_bfloat16* const conv_slot = p.conv + slot * p.conv_slot_stride;
      conv_slot[ch0 * 3 + 0] = cs0[1];
      conv_slot[ch0 * 3 + 1] = cs0[2];
      conv_slot[ch0 * 3 + 2] = x0;
      conv_slot[ch1 * 3 + 0] = cs1[1];
      conv_slot[ch1 * 3 + 1] = cs1[2];
      conv_slot[ch1 * 3 + 2] = x1;
      if constexpr (kDebug) {
        float* d = p.dbg + n * (kQKV + kHV * kDim);
        d[ch0] = o0;
        d[ch1] = o1;
      }
    }
    if (t < 2) {
      const float x = a_g + dtb_g;
      const float sp = x <= 20.f ? log1pf(expf(x)) : x;
      const float gg = -expf(alog_g) * sp;
      const float Gl = l > 0 ? *reinterpret_cast<const float*>(st + Stage::kG + t * 96 + (l - 1) * 4) : 0.f;
      const float Gn = Gl + gg;
      sm.gate[t][0] = Gn;
      sm.gate[t][1] = expf(Gn);
      sm.gate[t][2] = bf16_round(1.f / (1.f + expf(-b_g)));
    }
    if (t < 32) *reinterpret_cast<uint4*>(&sm.Br[0][0][0][0] + t * 8) = make_uint4(0, 0, 0, 0);
    device::ptx::bar_sync(bar_grp, 256);
    if (t == 0) {
      device::ptx::fence_proxy_async_shared();
      device::ptx::mbar_arrive(&S.empty[s]);
    }

    if constexpr (kRefit == REFIT_SHELL) {
      const uint32_t h = h_row, v = v_row;
      float acc = sm.x[t] + sm.x[256 + t] + sm.sc[h][0][v] + sm.sc[h][1][v] + sm.g[h][v % 20] + sm.zz[t] + S.nw[v];
      acc += __half2float(sm.ur[h][v % 24][v]) + __half2float(sm.ad[v % 28][v]) + sm.gate[h][0];
      const uint32_t* zw = reinterpret_cast<const uint32_t*>(&zr[0][0][0]);
#pragma unroll
      for (int i = 0; i < 32; ++i)
        acc += float(zw[i] & 0xff);
      *reinterpret_cast<__half*>(reg0 + h * gdn_lq::kHeadBytes + L::kU + (l * kDim + v) * 2) = __float2half_rn(acc);
      if (h == 0) *reinterpret_cast<__half*>(reg0 + L::kKh + (l * kDim + v) * 2) = __float2half_rn(acc);
      if (v == 0) *reinterpret_cast<float*>(reg0 + h * gdn_lq::kHeadBytes + L::kG + l * 4) = acc;
      p.out[n * (kHV * kDim) + (2 * hk + h) * kDim + v] = __float2bfloat16_rn(acc);
      if (t < 2) *reinterpret_cast<int32_t*>(reg0 + t * gdn_lq::kHeadBytes + L::kL) = (l + 1) >= lim ? 0 : l + 1;
      device::ptx::bar_sync(bar_grp, 256);
      continue;
    }

    // ---- prep ----
    float rq, rk;
    {
      const float4 qv = *reinterpret_cast<const float4*>(&sm.x[lane * 4]);
      const float4 kv = *reinterpret_cast<const float4*>(&sm.x[128 + lane * 4]);
      const float sq = device::warp::reduce_sum(qv.x * qv.x + qv.y * qv.y + qv.z * qv.z + qv.w * qv.w);
      const float sk = device::warp::reduce_sum(kv.x * kv.x + kv.y * kv.y + kv.z * kv.z + kv.w * kv.w);
      rq = (1.f / sqrtf(sq + 1e-6f)) * 0.08838834764831845f;
      rk = 1.f / sqrtf(sk + 1e-6f);
    }
    const float Gn_row = sm.gate[h_row][0];
    const float eG_row = sm.gate[h_row][1];
    const float beta_row = sm.gate[h_row][2];
    {
      const uint32_t i = v_row;
      const float c_col = sm.sc[h_row][1][i];
      const __half khh = __float2half_rn(sm.x[128 + i] * rk);
      const float qh = sm.x[i] * rq;
      const float cx_k = c_col * (eG_row * __half2float(khh));
      const float cx_q = c_col * (eG_row * qh);
      const __half kh_hi = __float2half_rn(cx_k), qh_hi = __float2half_rn(cx_q);
      const __half kh_lo = __float2half_rn(cx_k - __half2float(kh_hi));
      const __half qh_lo = __float2half_rn(cx_q - __half2float(qh_hi));
      const uint32_t qq = i / 32, kk = (i % 32) / 4, sub = i % 4;
      sm.Bz[h_row][kk][0 * 4 + qq][sub] = kh_hi;
      sm.Bz[h_row][kk][1 * 4 + qq][sub] = qh_hi;
      sm.Bz[h_row][kk][2 * 4 + qq][sub] = kh_lo;
      sm.Bz[h_row][kk][3 * 4 + qq][sub] = qh_lo;
      if (h_row == 0) {
        const float qs = qh * kDotScale;
        const __half q_hi = __float2half_rn(qs);
        const __half q_lo = __float2half_rn(qs - __half2float(q_hi));
        __half* bd = &sm.Bd[i / 16][0][0];
        bd[bfrag_idx(i, 0)] = khh;
        bd[bfrag_idx(i, 1)] = q_hi;
        bd[bfrag_idx(i, 2)] = q_lo;
        bd[bfrag_idx(i, 3)] = __float2half_rn(0.f);
        sm.ad[28][i] = khh;
        *reinterpret_cast<__half*>(reg0 + L::kKh + (l * kDim + i) * 2) = khh;
      }
    }
    device::ptx::bar_sync(bar_grp, 256);

    if (warp < 2) {  // dots -> record B
      float d[4] = {0.f, 0.f, 0.f, 0.f};
      const uint32_t mrow = warp * 16 + (lane % 8) + ((lane / 8) % 2) * 8;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t a[4];
        const __half* arow = mrow < kC + 9 ? &sm.ad[mrow][0] : &S.zero_row[0];
        ldsm_x4(a, arow + ks * 16 + (lane / 16) * 8);
        const uint2 b = (lane < 16) ? *reinterpret_cast<const uint2*>(&sm.Bd[ks][lane][0]) : make_uint2(0, 0);
        mma_16816(d, a, b.x, b.y);
      }
      const float lo0 = __shfl_xor_sync(0xffffffffu, d[0], 1);
      const float lo1 = __shfl_xor_sync(0xffffffffu, d[2], 1);
      if (lane % 4 == 0) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const uint32_t j = warp * 16 + lane / 4 + 8 * e;
          const float dk = e ? d[2] : d[0];
          const float dq = ((e ? d[3] : d[1]) + (e ? lo1 : lo0)) * (1.f / kDotScale);
          if (j < kC) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              float w = 0.f;
              if (int32_t(j) < l) w = expf(sm.gate[h][0] - sm.g[h][j]);
              const float wk = dk * w * kRecScale, wq = dq * w * kRecScale;
              __half* br = &sm.Br[h][j / 16][0][0];
              const __half wk_hi = __float2half_rn(wk), wq_hi = __float2half_rn(wq);
              br[bfrag_idx(j, 0)] = wk_hi;
              br[bfrag_idx(j, 1)] = wq_hi;
              br[bfrag_idx(j, 2)] = __float2half_rn(wk - __half2float(wk_hi));
              br[bfrag_idx(j, 3)] = __float2half_rn(wq - __half2float(wq_hi));
            }
          } else if (j < 28) {
            const uint32_t h = (j - 20) / 4, kq = 20 + (j - 20) % 4;
            const float tk = sm.gate[h][1] * dk * kRecScale, tq = sm.gate[h][1] * dq * kRecScale;
            __half* br = &sm.Br[h][1][0][0];
            const __half tk_hi = __float2half_rn(tk), tq_hi = __float2half_rn(tq);
            br[bfrag_idx(kq, 0)] = tk_hi;
            br[bfrag_idx(kq, 1)] = tq_hi;
            br[bfrag_idx(kq, 2)] = __float2half_rn(tk - __half2float(tk_hi));
            br[bfrag_idx(kq, 3)] = __float2half_rn(tq - __half2float(tq_hi));
          } else if (j == 28) {
            sm.qk = dq;
          }
        }
      }
    }

    float dz[2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
      for (int i = 0; i < 4; ++i)
        dz[mt][i] = 0.f;
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        const uint4& h0 = zr[mt][0][kk / 4];
        const uint4& h1 = zr[mt][1][kk / 4];
        const uint32_t wr = (kk % 4 == 0) ? h0.x : (kk % 4 == 1) ? h0.y : (kk % 4 == 2) ? h0.z : h0.w;
        const uint32_t wr8 = (kk % 4 == 0) ? h1.x : (kk % 4 == 1) ? h1.y : (kk % 4 == 2) ? h1.z : h1.w;
        uint32_t a[4];
        i8x4_to_f16x2(wr, a[0], a[2]);
        i8x4_to_f16x2(wr8, a[1], a[3]);
        uint2 b = make_uint2(0, 0);
        if (r0 < 4) b = *reinterpret_cast<const uint2*>(&sm.Bz[mh][kk][lane][0]);
        mma_16816(dz[mt], a, b.x, b.y);
      }
    }
    device::ptx::bar_sync(bar_grp, 256);

    float dr[2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
      for (int i = 0; i < 4; ++i)
        dr[mt][i] = 0.f;
      const uint32_t m0 = rb + mt * 16;
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        uint32_t a[4];
        const uint32_t kr = ks * 16 + (lane % 8) + (lane / 16) * 8;
        const __half* arow = kr < kC + 4 ? &sm.ur[mh][kr][0] : &S.zero_row[0];
        ldsm_x4_trans(a, arow + m0 + ((lane / 8) % 2) * 8);
        const uint2 b = (lane < 16) ? *reinterpret_cast<const uint2*>(&sm.Br[mh][ks][lane][0]) : make_uint2(0, 0);
        mma_16816(dr[mt], a, b.x, b.y);
      }
    }

    const float qk = sm.qk;
    float wv[2][2], ov[2][2];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const uint32_t R = rb + mt * 16 + r0 + 8 * e;
        const float s_v = sm.sc[mh][0][R];
        const float pw = s_v * dz[mt][2 * e] + dr[mt][2 * e] * (1.f / kRecScale);  // q4 0: hi, q4 1: lo parts
        const float po = s_v * dz[mt][2 * e + 1] + dr[mt][2 * e + 1] * (1.f / kRecScale);
        wv[mt][e] = pw + __shfl_xor_sync(0xffffffffu, pw, 1);
        ov[mt][e] = po + __shfl_xor_sync(0xffffffffu, po, 1);
      }
    }
    // lane L takes row rb + L: (m-tile, half, r0) = (L / 16, (L / 8) % 2, L % 8), held by lane 4 (L % 8)
    float wsum = 0.f, osum = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float a = __shfl_sync(0xffffffffu, wv[i / 2][i % 2], 4 * (lane % 8));
      const float b = __shfl_sync(0xffffffffu, ov[i / 2][i % 2], 4 * (lane % 8));
      if (lane / 8 == uint32_t(i)) wsum = a, osum = b;
    }
    const uint32_t R = rb + lane;
    const float U = beta_row * (sm.x[256 + mh * 128 + R] - wsum);
    const float o = osum + qk * U;
    *reinterpret_cast<__half*>(reg0 + mh * gdn_lq::kHeadBytes + L::kU + (l * kDim + R) * 2) = __float2half_rn(U);
    if constexpr (kDebug) p.dbg[n * (kQKV + kHV * kDim) + kQKV + (2 * hk + mh) * kDim + R] = o;
    if (t % 128 == 0) *reinterpret_cast<float*>(reg0 + mh * gdn_lq::kHeadBytes + L::kG + l * 4) = Gn_row;
    const float ss = device::warp::reduce_sum(o * o);
    if (lane == 0) sm.red[warp] = ss;
    device::ptx::bar_sync(bar_grp + 1 + mh, 128);
    const float tot = sm.red[4 * mh] + sm.red[4 * mh + 1] + sm.red[4 * mh + 2] + sm.red[4 * mh + 3];
    const float rn = rsqrtf(tot * (1.f / kDim) + p.eps);
    const float zf = sm.zz[mh * 128 + R];
    const float y = o * rn * S.nw[R] * (zf / (1.f + expf(-zf)));
    p.out[n * (kHV * kDim) + (2 * hk + mh) * kDim + R] = __float2bfloat16_rn(y);
    if (t < 2) {
      int32_t ln = l + 1;
      if constexpr (kRefit == REFIT_WRAP) ln = ln >= lim ? 0 : ln;
      *reinterpret_cast<int32_t*>(reg0 + t * gdn_lq::kHeadBytes + L::kL) = ln;
    }
    if constexpr (kRefit == REFIT_LIST) {
      if (t == 0 && l + 1 >= lim) p.due[2 + atomicAdd(&p.due[0], 1)] = static_cast<int32_t>(n * kHK + hk);
    }
    device::ptx::bar_sync(bar_grp, 256);  // group buffers are rewritten by the next unit
  }
}

// ================================================================== decode flush (R-flush, list form)
// Launched after the step kernel with PDL (no trigger in the step: the flush is released at the step's completion).
// The step appended every (request, key head) whose window filled to p.due; each CTA refits value heads of the
// list (R's RefitCore, cold1), writes l = 0, lim = P to the refitted head's copy, and the last CTA to finish
// resets the list. With nothing due every CTA exits after one load.
__global__ void __launch_bounds__(256, 2) decode_flush_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  device::PDLWaitPrimary<true>();
  const uint32_t count = static_cast<uint32_t>(*reinterpret_cast<volatile int32_t*>(&p.due[0]));
  const uint32_t heads = 2 * count;
  auto& rs = *reinterpret_cast<typename RefitCoreCold1::Smem*>(smem_raw);
  for (uint32_t u = blockIdx.x; u < heads; u += gridDim.x) {
    const uint32_t unit = static_cast<uint32_t>(p.due[2 + u / 2]);
    const uint32_t n = unit / kHK, hk = unit % kHK, h = u % 2;
    const int32_t slot = p.idx[n];
    uint8_t* reg0 = p.pool + slot * p.slot_bytes + (2 * hk) * gdn_lq::kHeadBytes;
    uint8_t* hs = reg0 + h * gdn_lq::kHeadBytes;
    const int32_t lnew = *reinterpret_cast<const int32_t*>(hs + L::kL);
    __syncthreads();  // smem reuse across units
    RefitCoreCold1 core;
    core.run(rs, hs, reg0, hs, static_cast<uint32_t>(lnew), p.q0, 1e-6f);
    if (threadIdx.x == 0) *reinterpret_cast<int2*>(hs + L::kL) = make_int2(0, int32_t(kP));
  }
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(&p.due[1], 1) == int32_t(gridDim.x) - 1) {  // every CTA has read the count
      p.due[0] = 0;
      p.due[1] = 0;
    }
  }
}

}  // namespace gdn_lq_dec

template <bool kDebug, int kRefit>
struct GdnLqDecode {
  static int64_t fill(
      gdn_lq_dec::Params& p,
      DLDevice& dev,
      tvm::ffi::TensorView qkvz,
      tvm::ffi::TensorView ba,
      tvm::ffi::TensorView idx,
      tvm::ffi::TensorView pool,
      tvm::ffi::TensorView conv,
      tvm::ffi::TensorView conv_w,
      tvm::ffi::TensorView A_log,
      tvm::ffi::TensorView dt_bias,
      tvm::ffi::TensorView norm_w,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView dbg,
      double eps) {
    using namespace host;
    using namespace gdn_lq_dec;
    auto N = SymbolicSize{"N"};
    auto S = SymbolicSize{"slots"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({N, -1}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(qkvz);
    TensorMatcher({N, -1}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(ba);
    TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(idx);
    TensorMatcher({S, kHV, kDim, kDim})
        .with_strides({-1, kDim * kDim, kDim, 1})
        .with_dtype<fp32_t>()
        .with_device<kDLCUDA>(device)
        .verify(pool);
    TensorMatcher({S, kQKV, 3}).with_strides({-1, 3, 1}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(conv);
    TensorMatcher({kQKV, 4}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(conv_w);
    TensorMatcher({kHV}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(A_log);
    TensorMatcher({kHV}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(dt_bias);
    TensorMatcher({kDim}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(norm_w);
    TensorMatcher({N, kHV * kDim}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(out);
    CHECK_HOST(qkvz.size(1) >= int64_t(kQKV + kHV * kDim)) << "qkvz width";
    CHECK_HOST(ba.size(1) >= int64_t(2 * kHV)) << "ba width";
    if constexpr (kDebug) {
      TensorMatcher({N, kQKV + kHV * kDim}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(dbg);
    }
    const int64_t n = N.unwrap();
    dev = device.unwrap();
    p.qkvz = static_cast<const __nv_bfloat16*>(qkvz.data_ptr());
    p.ba = static_cast<const __nv_bfloat16*>(ba.data_ptr());
    p.idx = static_cast<const int32_t*>(idx.data_ptr());
    p.pool = static_cast<uint8_t*>(pool.data_ptr());
    p.conv = static_cast<__nv_bfloat16*>(conv.data_ptr());
    p.conv_w = static_cast<const __nv_bfloat16*>(conv_w.data_ptr());
    p.A_log = static_cast<const float*>(A_log.data_ptr());
    p.dt_bias = static_cast<const __nv_bfloat16*>(dt_bias.data_ptr());
    p.norm_w = static_cast<const __nv_bfloat16*>(norm_w.data_ptr());
    p.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    p.dbg = kDebug ? static_cast<float*>(dbg.data_ptr()) : nullptr;
    p.qkvz_stride = qkvz.stride(0);
    p.ba_stride = ba.stride(0);
    p.slot_bytes = pool.stride(0) * 4;
    p.conv_slot_stride = conv.stride(0);
    p.eps = static_cast<float>(eps);
    return n;
  }
  static void
  run(tvm::ffi::TensorView qkvz,
      tvm::ffi::TensorView ba,
      tvm::ffi::TensorView idx,
      tvm::ffi::TensorView pool,
      tvm::ffi::TensorView conv,
      tvm::ffi::TensorView conv_w,
      tvm::ffi::TensorView A_log,
      tvm::ffi::TensorView dt_bias,
      tvm::ffi::TensorView norm_w,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView dbg,
      double eps,
      tvm::ffi::TensorView q0) {
    using namespace host;
    using namespace gdn_lq_dec;
    Params p{};
    DLDevice dev;
    const int64_t n = fill(p, dev, qkvz, ba, idx, pool, conv, conv_w, A_log, dt_bias, norm_w, out, dbg, eps);
    if (n == 0) return;
    if constexpr (kRefit == REFIT_LIST) p.due = static_cast<int32_t*>(q0.data_ptr());  // q0 slot carries the list
    if constexpr (kRefit == REFIT_FUSED) {
      auto qdev = SymbolicDevice{};
      qdev.set_options<kDLCUDA>();
      TensorMatcher({128, 4}).with_dtype<fp32_t>().with_device<kDLCUDA>(qdev).verify(q0);
      p.q0 = static_cast<const float*>(q0.data_ptr());
    }
    constexpr size_t kSmem = kRefit == REFIT_FUSED && sizeof(typename RefitCoreCold1::Smem) > sizeof(Smem)
                                 ? sizeof(typename RefitCoreCold1::Smem)
                                 : sizeof(Smem);
    constexpr auto kernel = decode_kernel<kDebug, kRefit>;
    runtime::set_max_dynamic_smem_per_device<kernel>(dev, kSmem);
    LaunchKernel(dim3(kHK, static_cast<uint32_t>(n)), 256, dev, kSmem)(kernel, p);
  }
};

template <uint32_t kGroups, uint32_t kStages, bool kDebug, int kRefit>
struct GdnLqDecodePersistent {
  static void
  run(tvm::ffi::TensorView qkvz,
      tvm::ffi::TensorView ba,
      tvm::ffi::TensorView idx,
      tvm::ffi::TensorView pool,
      tvm::ffi::TensorView conv,
      tvm::ffi::TensorView conv_w,
      tvm::ffi::TensorView A_log,
      tvm::ffi::TensorView dt_bias,
      tvm::ffi::TensorView norm_w,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView dbg,
      double eps,
      tvm::ffi::TensorView q0) {
    using namespace host;
    using namespace gdn_lq_dec;
    Params p{};
    DLDevice dev;
    const int64_t n = GdnLqDecode<kDebug, kRefit>::fill(
        p, dev, qkvz, ba, idx, pool, conv, conv_w, A_log, dt_bias, norm_w, out, dbg, eps);
    if (n == 0) return;
    if constexpr (kRefit == REFIT_LIST) p.due = static_cast<int32_t*>(q0.data_ptr());
    using SM = PersistentSmem<kGroups, kStages>;
    constexpr auto kernel = decode_persistent_kernel<kGroups, kStages, kDebug, kRefit>;
    runtime::set_max_dynamic_smem_per_device<kernel>(dev, sizeof(SM));
    const uint32_t units = static_cast<uint32_t>(n) * kHK;
    const uint32_t sms = runtime::get_sm_count(dev.device_id);
    const uint32_t grid = units < sms ? units : sms;
    LaunchKernel(grid, kGroups * 256 + 32, dev, sizeof(SM))(kernel, p, units);
  }
};

struct GdnLqDecodeFlush {
  // pool [S, 32, 128, 128] fp32 format view; idx [N] int32; due int32 [2 + 16 N]; q0 [128, 4] fp32
  static void
  run(tvm::ffi::TensorView pool,
      tvm::ffi::TensorView idx,
      tvm::ffi::TensorView due,
      tvm::ffi::TensorView q0,
      bool pdl) {
    using namespace host;
    using namespace gdn_lq_dec;
    auto N = SymbolicSize{"N"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({-1, kHV, kDim, kDim})
        .with_strides({-1, kDim * kDim, kDim, 1})
        .with_dtype<fp32_t>()
        .with_device<kDLCUDA>(device)
        .verify(pool);
    TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(idx);
    TensorMatcher({128, 4}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(q0);
    const int64_t n = N.unwrap();
    TensorMatcher({2 + 16 * n}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(due);
    if (n == 0) return;
    Params p{};
    p.pool = static_cast<uint8_t*>(pool.data_ptr());
    p.idx = static_cast<const int32_t*>(idx.data_ptr());
    p.slot_bytes = pool.stride(0) * 4;
    p.q0 = static_cast<const float*>(q0.data_ptr());
    p.due = static_cast<int32_t*>(due.data_ptr());
    const DLDevice dev = device.unwrap();
    constexpr size_t kSmem = sizeof(typename RefitCoreCold1::Smem);
    runtime::set_max_dynamic_smem_per_device<decode_flush_kernel>(dev, kSmem);
    const uint32_t sms = runtime::get_sm_count(dev.device_id);
    const uint32_t max_heads = static_cast<uint32_t>(32 * n);
    const uint32_t grid = max_heads < 2 * sms ? max_heads : 2 * sms;
    LaunchKernel(grid, 256, dev, kSmem).enable_pdl(pdl)(decode_flush_kernel, p);
  }
};

}  // namespace
}  // namespace sglang
