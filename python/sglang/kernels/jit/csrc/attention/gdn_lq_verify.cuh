// LeapQuant GDN target verify (Qwen3.5: H_k 16, H_v 32, K = V = 128, T = 4 chain drafts), fused:
// unpack of qkvz / ba + causal conv (width 4, SiLU) with the production rollback windows + gates + l2norm
// + LeapQuant window (parallel WY form) + gated RMSNorm. One CTA (256 threads) per (request, key head) = 2 value heads.
// Reads the base and the committed records, writes the 4 draft records (u per value head, k^ per key head, G),
// the conv windows (production layout, so the production commit / rollback is reused) and the normed output.
// Never writes the base, l, or the conv state (the production commit always overwrites the conv state from
// the window of the last accepted token).
//
// Window math per value head (see the GDN oracle, leapquant_ref.gdn_window): with x^_m in {k^_m, q^_m},
//   W0 / O0 [m][v] = s_v sum_k Z[v][k] fp16(c_k e^{G_m} x^_m[k]) + sum_j Ur[j][v] e^{G_m - G_j} <Kr_j, x^_m>
// (Ur = [U~; u_0..u_{l-1}], Kr = [K~; k^_0..k^_{l-1}], G_j = 0 for the 4 compensator rows),
//   U = (I + diag(beta) A)^{-1} diag(beta) (V - W0), O = O0 + Qk U.
#pragma once

#include "gdn_lq_refit.cuh"

namespace sglang {

namespace lqv {

using namespace lqr;

inline constexpr uint32_t kT = 4;
inline constexpr uint32_t kHK = 16, kHV = 32;
inline constexpr uint32_t kQKVZ = 12288, kBA = 64, kConv = 8192;
inline constexpr uint32_t kOffK = 2048, kOffV = 4096, kOffZ = 8192;

struct VerifyParams {
  const bf16_t* __restrict__ qkvz;             // [N * T, 12288]
  const bf16_t* __restrict__ ba;               // [N * T, 64]
  const int32_t* __restrict__ slot_idx;        // [N]
  const int32_t* __restrict__ row_idx;         // [N] intermediate window row
  uint8_t* __restrict__ pool;                  // [S, HV, kHeadBytes] (one layer)
  const bf16_t* __restrict__ conv_state;       // [S, 8192, 3]
  const bf16_t* __restrict__ conv_w;           // [8192, 4]
  bf16_t* __restrict__ conv_win;               // [rows, 8192, T + 2]
  const float* __restrict__ A_log;             // [HV]
  const float* __restrict__ dt_bias;           // [HV]
  const bf16_t* __restrict__ norm_w;           // [V]
  bf16_t* __restrict__ out;                    // [N * T, HV * V]
  int64_t conv_state_stride, conv_win_stride;  // elements per slot / row
  float eps;
};

}  // namespace lqv

template <uint32_t kC, bool kShell, uint32_t kHPC>
struct LQVerify {
  static_assert(kHPC == 1 || kHPC == 2, "value heads per CTA");
  using L = gdn_lq::Layout<kC>;
  static constexpr uint32_t kDim = gdn_lq::kDim, kRank = gdn_lq::kRank, kHeadBytes = gdn_lq::kHeadBytes;
  static constexpr uint32_t kThreads = 128 * kHPC, kWarps = 4 * kHPC, kRows = 32, kHP = kDim + 8;
  static constexpr uint32_t kNch = 2 * kDim + kHPC * kDim;  // conv channels: q, k, v of the CTA's heads
  static constexpr uint32_t kXP = 4 * 48 + 8;               // padded f16 rows, xcol(k) = k + 16 (k / 32)
  static_assert(kRank + kC <= kRows);
  SGL_DEVICE static uint32_t xcol(uint32_t k) {
    return k + 16 * (k / 32);
  }

  struct Smem {
    union {
      struct {
        alignas(16) bf16_t xq[lqv::kT][kDim];  // raw (pre-conv) q, k, v (2 heads)
        alignas(16) bf16_t xk[lqv::kT][kDim];
        alignas(16) bf16_t xv[lqv::kT][kHPC * kDim];
        alignas(16) bf16_t cw[kNch][4];    // conv weights of the 512 channels
        alignas(16) bf16_t cst[kNch * 3];  // conv state of the 512 channels [ch][3]
      } in;                                // dead after the conv
      alignas(16) __half xh[16][kXP];      // dots B operand: [k^ | q^ 2^8 hi] [0 | q^ 2^8 lo] (l2norm .. dots)
      struct {
        alignas(16) float rowx[kWarps][32][8];  // per-warp row exchange of the main products
        alignas(16) bf16_t outs[lqv::kT][kHPC * kDim];
        alignas(16) __half us[kHPC][lqv::kT][kDim];
      } fin;  // after the main products
    } u1;
    union {
      struct {
        alignas(16) float qf[lqv::kT][kDim];  // post-conv q, k (bf16-rounded values)
        alignas(16) float kf[lqv::kT][kDim];
      } qk;                                       // conv .. l2norm
      alignas(16) float dots[kHPC][2][kRows][8];  // [head][hi / lo n-tile][j][n]
    } u2;
    alignas(16) bf16_t xz[lqv::kT][kHPC * kDim];
    alignas(16) float vf[lqv::kT][kHPC * kDim];  // post-conv v
    alignas(16) float kd[lqv::kT][kDim];         // k^ (fp16-rounded value)
    alignas(16) float qd[lqv::kT][kDim];         // q^ (scaled by K^-1/2)
    alignas(16) __half ur[kHPC][kRows][kHP];     // [head][j][v]
    alignas(16) __half kr[kHPC][kRows][kHP];     // [head][j][k]
    alignas(16) __half xs[kHPC][8][kXP];         // base operand fp16(c e^{G_m} x^_m)
    alignas(16) __half bl[kHPC][16][kRows + 8];  // coef 2^13 [hi 0..7 | lo 8..15][j]
    alignas(16) float c[kHPC][kDim];
    alignas(16) float G[kHPC][kC];
    alignas(16) float wrec[kHPC][kRows][lqv::kT];  // e^{G_m - G_j} (0 beyond the live rows)
    alignas(16) float Gm[kHPC][lqv::kT], eGm[kHPC][lqv::kT], beta[kHPC][lqv::kT], gate[kHPC][lqv::kT];
    alignas(16) float rawA[lqv::kT][lqv::kT], rawQ[lqv::kT][lqv::kT];
    alignas(16) float Pm[kHPC][lqv::kT][lqv::kT], Qk[kHPC][lqv::kT][lqv::kT];
    alignas(16) float nsum[kHPC][4][lqv::kT];
    alignas(16) float nw[kDim];
    alignas(16) __half khs[lqv::kT][kDim];
  };

  SGL_DEVICE static void
  store(Smem& sm, const lqv::VerifyParams& p, uint32_t b, uint32_t h0, const uint8_t* lbase, int32_t l, uint32_t tid) {
    // chunks of 16 B: out rows 16 kHPC per token (64 kHPC), u records 64 per head (64 kHPC)
    using namespace lqv;
    // stores: outputs, records u (1 KB per head: rows l .. l + 3 are contiguous), k^ (even head), G
    {
      const uint32_t i = tid;
      if (i < 64 * kHPC) {
        const uint32_t m = i / (16 * kHPC), c = i % (16 * kHPC);
        *reinterpret_cast<uint4*>(p.out + static_cast<int64_t>(b * kT + m) * (kHV * kDim) + h0 * kDim + c * 8) =
            *reinterpret_cast<const uint4*>(&sm.u1.fin.outs[m][c * 8]);
      } else {
        const uint32_t i2 = i - 64 * kHPC, hh = i2 / 64, c = i2 % 64;
        uint8_t* dst = const_cast<uint8_t*>(lbase) + static_cast<int64_t>(h0 + hh) * kHeadBytes + L::kU + l * 256;
        reinterpret_cast<uint4*>(dst)[c] = reinterpret_cast<const uint4*>(&sm.u1.fin.us[hh][0][0])[c];
      }
      if ((h0 & 1) == 0 && tid < 64) {
        uint8_t* dst = const_cast<uint8_t*>(lbase) + static_cast<int64_t>(h0) * kHeadBytes + L::kKh + l * 256;
        reinterpret_cast<uint4*>(dst)[tid] = reinterpret_cast<const uint4*>(&sm.khs[0][0])[tid];
      } else if (tid >= 64 && tid < 64 + kHPC * kT) {
        const uint32_t hh = (tid - 64) / kT, m = (tid - 64) % kT;
        reinterpret_cast<float*>(
            const_cast<uint8_t*>(lbase) + static_cast<int64_t>(h0 + hh) * kHeadBytes + L::kG)[l + m] = sm.Gm[hh][m];
      }
    }
  }

  SGL_DEVICE static void run(Smem& sm, const lqv::VerifyParams& p) {
    using namespace lqv;
    using namespace lqr;
    const uint32_t tid = threadIdx.x, lane = tid % 32, warp = __shfl_sync(0xffffffffu, tid / 32, 0);
    const uint32_t g = lane / 4, t = lane % 4;
    const uint32_t b = blockIdx.x / (kHV / kHPC), h0 = (blockIdx.x % (kHV / kHPC)) * kHPC, kh = h0 / 2;
    const int32_t slot = p.slot_idx[b];
    const int32_t row = p.row_idx[b];
    // ---- inputs (independent of the slot)
    {
      // per token: q 16 chunks, k 16, v 16 kHPC, z 16 kHPC
      constexpr uint32_t kCh = 32 + 32 * kHPC;
      for (uint32_t i = tid; i < kT * kCh; i += kThreads) {
        const uint32_t m = i / kCh, c = i % kCh;
        const bf16_t* src = p.qkvz + static_cast<int64_t>(b * kT + m) * kQKVZ;
        void* dst;
        if (c < 16) {
          src += kh * kDim + c * 8, dst = &sm.u1.in.xq[m][c * 8];
        } else if (c < 32) {
          src += kOffK + kh * kDim + (c - 16) * 8, dst = &sm.u1.in.xk[m][(c - 16) * 8];
        } else if (c < 32 + 16 * kHPC) {
          src += kOffV + h0 * kDim + (c - 32) * 8, dst = &sm.u1.in.xv[m][(c - 32) * 8];
        } else {
          src += kOffZ + h0 * kDim + (c - 32 - 16 * kHPC) * 8, dst = &sm.xz[m][(c - 32 - 16 * kHPC) * 8];
        }
        cp_async_16_zfill(dst, src, true);
      }
      // conv weights: q 128 ch, k 128 ch, v 256 ch, 8 B each
      for (uint32_t i = tid; i < kNch * 8 / 16; i += kThreads) {
        const uint32_t ch = i * 2;  // two channels per 16 B
        const uint32_t gch = ch < kDim       ? kh * kDim + ch
                             : ch < 2 * kDim ? kOffK + kh * kDim + ch - kDim
                                             : kOffV + h0 * kDim + ch - 2 * kDim;
        cp_async_16_zfill(&sm.u1.in.cw[ch][0], p.conv_w + static_cast<int64_t>(gch) * 4, true);
      }
      if (tid < kDim / 4) {
        const uint2 w4 = reinterpret_cast<const uint2*>(p.norm_w)[tid];
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w4.x));
        const float2 c = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w4.y));
        *reinterpret_cast<float4*>(&sm.nw[4 * tid]) = make_float4(a.x, a.y, c.x, c.y);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    if (slot < 0) return;  // padding row (block-uniform)
    // ---- slot-dependent: conv state, LeapQuant state (records speculatively for the whole capacity)
    const uint8_t* lbase = p.pool + static_cast<int64_t>(slot) * kHV * kHeadBytes;
    const int32_t l = *reinterpret_cast<const int32_t*>(lbase + static_cast<int64_t>(h0) * kHeadBytes + L::kL);
    {
      const bf16_t* cs = p.conv_state + static_cast<int64_t>(slot) * p.conv_state_stride;
      // q block 768 B, k block 768 B, v block 1536 B
      for (uint32_t i = tid; i < 96 + 48 * kHPC; i += kThreads) {
        const bf16_t* src = i < 48   ? cs + kh * kDim * 3 + i * 8
                            : i < 96 ? cs + (kOffK + kh * kDim) * 3 + (i - 48) * 8
                                     : cs + (kOffV + h0 * kDim) * 3 + (i - 96) * 8;
        cp_async_16_zfill(&sm.u1.in.cst[i * 8], src, true);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    {
      for (uint32_t i = tid; i < kHPC * kRows * 16; i += kThreads) {
        const uint32_t hh = i / (kRows * 16), j = (i / 16) % kRows, cc = i % 16;
        const uint8_t* hs = lbase + static_cast<int64_t>(h0 + hh) * kHeadBytes;
        const uint8_t* ks = lbase + static_cast<int64_t>(2 * kh) * kHeadBytes;
        const bool valid = j < kRank + kC;
        const uint8_t* su = j < kRank ? hs + L::kUt + j * 256 : hs + L::kU + (j - kRank) * 256;
        const uint8_t* sk = j < kRank ? hs + L::kKt + j * 256 : ks + L::kKh + (j - kRank) * 256;
        cp_async_16_zfill(&sm.ur[hh][j][cc * 8], valid ? su + cc * 16 : hs, valid);
        cp_async_16_zfill(&sm.kr[hh][j][cc * 8], valid ? sk + cc * 16 : hs, valid);
      }
      if (tid < 32 * kHPC) {
        const uint32_t hh = tid / 32, c4 = tid % 32;
        cp_async_16_zfill(
            &sm.c[hh][c4 * 4], lbase + static_cast<int64_t>(h0 + hh) * kHeadBytes + L::kCc + c4 * 16, true);
      } else if (tid >= 64 && tid < 64 + kHPC * (kC / 4)) {
        const uint32_t i = tid - 64, hh = i / (kC / 4), c4 = i % (kC / 4);
        cp_async_16_zfill(
            &sm.G[hh][c4 * 4], lbase + static_cast<int64_t>(h0 + hh) * kHeadBytes + L::kG + c4 * 16, true);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    // Z rows of this warp (head hw, rows 32 rbw .. + 32) straight into registers; s of the 4 rows
    const uint32_t hw = warp >> 2, rbw = warp & 3;
    const uint8_t* hsw = lbase + static_cast<int64_t>(h0 + hw) * kHeadBytes;
    uint32_t zw[2][2][8];  // [m-tile][row g / g + 8][word]
    float srow[2][2];
#pragma unroll
    for (uint32_t mt = 0; mt < 2; ++mt)
#pragma unroll
      for (uint32_t hr = 0; hr < 2; ++hr) {
        const uint32_t v = 32 * rbw + 16 * mt + g + 8 * hr;
        device::AlignedVector<uint32_t, 8> zv;
        zv.load(hsw + L::kZ + v * kDim + 32 * t);
#pragma unroll
        for (int i = 0; i < 8; ++i)
          zw[mt][hr][i] = zv[i];
        srow[mt][hr] = reinterpret_cast<const float*>(hsw + L::kS)[v];
      }
    // gate inputs (raw bf16 a, b) for the two heads x 4 tokens
    float a_in = 0.f, b_in = 0.f, a_log_h = 0.f, dtb = 0.f;
    if (tid < kHPC * kT) {
      const uint32_t hh = tid / kT, m = tid % kT;
      const bf16_t* bar = p.ba + static_cast<int64_t>(b * kT + m) * kBA;
      b_in = static_cast<float>(bar[h0 + hh]);
      a_in = static_cast<float>(bar[kHV + h0 + hh]);
      a_log_h = p.A_log[h0 + hh];
      dtb = p.dt_bias[h0 + hh];
    }
    // ---- conv (inputs + conv state), windows
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    __syncthreads();
    bf16_t* win = p.conv_win + static_cast<int64_t>(row) * p.conv_win_stride;
#pragma unroll
    for (uint32_t r = 0; r < kNch / kThreads; ++r) {
      const uint32_t ch = tid + r * kThreads;
      const bf16_t* xr;
      uint32_t gch, col;
      if (ch < kDim) {
        xr = &sm.u1.in.xq[0][ch], col = ch, gch = kh * kDim + ch;
      } else if (ch < 2 * kDim) {
        xr = &sm.u1.in.xk[0][ch - kDim], col = ch - kDim, gch = kOffK + kh * kDim + ch - kDim;
      } else {
        xr = &sm.u1.in.xv[0][ch - 2 * kDim], col = ch - 2 * kDim, gch = kOffV + h0 * kDim + ch - 2 * kDim;
      }
      const uint32_t xstride = ch < 2 * kDim ? kDim : kHPC * kDim;
      float seq[3 + kT];
      bf16_t sb[3 + kT];
#pragma unroll
      for (int i = 0; i < 3; ++i)
        sb[i] = sm.u1.in.cst[ch * 3 + i];
#pragma unroll
      for (uint32_t m = 0; m < kT; ++m)
        sb[3 + m] = xr[m * xstride];
#pragma unroll
      for (int i = 0; i < 3 + static_cast<int>(kT); ++i)
        seq[i] = static_cast<float>(sb[i]);
      float w[4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        w[i] = static_cast<float>(sm.u1.in.cw[ch][i]);
#pragma unroll
      for (uint32_t m = 0; m < kT && !kShell; ++m) {
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i)
          acc = fmaf(w[i], seq[m + i], acc);
        const float y = static_cast<float>(static_cast<bf16_t>(acc / (1.f + __expf(-acc))));
        if (ch < kDim) {
          sm.u2.qk.qf[m][col] = y;
        } else if (ch < 2 * kDim) {
          sm.u2.qk.kf[m][col] = y;
        } else {
          sm.vf[m][col] = y;
        }
      }
      // window row [state1, state2, x0 .. x3]
      bf16_t* wr = win + static_cast<int64_t>(gch) * (kT + 2);
      __nv_bfloat162 w0, w1, w2;
      w0.x = sb[1], w0.y = sb[2], w1.x = sb[3], w1.y = sb[4], w2.x = sb[5], w2.y = sb[6];
      reinterpret_cast<__nv_bfloat162*>(wr)[0] = w0;
      reinterpret_cast<__nv_bfloat162*>(wr)[1] = w1;
      reinterpret_cast<__nv_bfloat162*>(wr)[2] = w2;
    }
    // gates
    if (tid < kHPC * kT) {
      const uint32_t hh = tid / kT, m = tid % kT;
      const float x = a_in + dtb;
      const float sp = x <= 20.f ? log1pf(expf(x)) : x;
      sm.gate[hh][m] = -expf(a_log_h) * sp;
      sm.beta[hh][m] = 1.f / (1.f + expf(-b_in));
    }
    __syncthreads();
    // ---- l2norm: warp w -> vector (q or k, token w % 4)
    if constexpr (!kShell)
      for (uint32_t vid = warp; vid < 8; vid += kWarps) {
        const uint32_t m = vid & 3;
        const bool isk = vid >= 4;
        const float* src = isk ? sm.u2.qk.kf[m] : sm.u2.qk.qf[m];
        const float4 x = *reinterpret_cast<const float4*>(&src[4 * lane]);
        float ss = x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
#pragma unroll
        for (uint32_t o = 16; o >= 1; o >>= 1)
          ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float r = rsqrtf(ss + 1e-6f);
        if (isk) {
          const uint32_t h01 = f2h2(x.x * r, x.y * r), h23 = f2h2(x.z * r, x.w * r);
          const float2 f01 = half2_to_float2(h01), f23 = half2_to_float2(h23);
          *reinterpret_cast<float4*>(&sm.kd[m][4 * lane]) = make_float4(f01.x, f01.y, f23.x, f23.y);
          *reinterpret_cast<uint2*>(&sm.khs[m][4 * lane]) = make_uint2(h01, h23);
          // dots operand rows 0..3 (exact), 8..11 zero
          const uint32_t k0 = 4 * lane;
          *reinterpret_cast<uint2*>(&sm.u1.xh[m][xcol(k0)]) = make_uint2(h01, h23);
          *reinterpret_cast<uint2*>(&sm.u1.xh[8 + m][xcol(k0)]) = make_uint2(0u, 0u);
        } else {
          const float sc = r * 0.08838834764831845f;  // K^-1/2
          const float4 qv = make_float4(x.x * sc, x.y * sc, x.z * sc, x.w * sc);
          *reinterpret_cast<float4*>(&sm.qd[m][4 * lane]) = qv;
          const uint32_t k0 = 4 * lane;
          *reinterpret_cast<uint2*>(&sm.u1.xh[4 + m][xcol(k0)]) =
              make_uint2(hilo2(qv.x * 256.f, qv.y * 256.f, false), hilo2(qv.z * 256.f, qv.w * 256.f, false));
          *reinterpret_cast<uint2*>(&sm.u1.xh[12 + m][xcol(k0)]) =
              make_uint2(hilo2(qv.x * 256.f, qv.y * 256.f, true), hilo2(qv.z * 256.f, qv.w * 256.f, true));
        }
      }
    __syncthreads();
    // ---- intra-draft raw dots: <k^_m, k^_j> (j < m), <q^_m, k^_j> (j <= m); warp w: pairs 2w, 2w + 1 of 16
    if constexpr (!kShell) {
#pragma unroll
      for (uint32_t pp = 0; pp < 16 / kWarps; ++pp) {
        const uint32_t id = (16 / kWarps) * warp + pp, m = id / 4, j = id % 4;
        const float4 kj = *reinterpret_cast<const float4*>(&sm.kd[j][4 * lane]);
        const float4 km = *reinterpret_cast<const float4*>(&sm.kd[m][4 * lane]);
        const float4 qm = *reinterpret_cast<const float4*>(&sm.qd[m][4 * lane]);
        float a = km.x * kj.x + km.y * kj.y + km.z * kj.z + km.w * kj.w;
        float q = qm.x * kj.x + qm.y * kj.y + qm.z * kj.z + qm.w * kj.w;
#pragma unroll
        for (uint32_t o = 16; o >= 1; o >>= 1) {
          a += __shfl_xor_sync(0xffffffffu, a, o);
          q += __shfl_xor_sync(0xffffffffu, q, o);
        }
        if (lane == 0) sm.rawA[m][j] = a, sm.rawQ[m][j] = q;
      }
    }
    // ---- state landed
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
    if constexpr (kShell) {
      // same loads and stores, no math: every loaded buffer feeds a stored value
      uint32_t fold = __float_as_uint(srow[0][0]) ^ __float_as_uint(srow[1][1]);
#pragma unroll
      for (uint32_t mt = 0; mt < 2; ++mt)
#pragma unroll
        for (uint32_t hr = 0; hr < 2; ++hr)
#pragma unroll
          for (int i = 0; i < 8; ++i)
            fold ^= zw[mt][hr][i];
      fold = (fold ^ __float_as_uint(sm.c[hw][lane]) ^ __float_as_uint(sm.G[hw][lane % kC])) & 1u;
      const uint32_t vrow = 32 * rbw + lane;
#pragma unroll
      for (uint32_t m = 0; m < kT; ++m) {
        sm.u1.fin.outs[m][hw * kDim + vrow] =
            __ushort_as_bfloat16(__bfloat16_as_ushort(sm.xz[m][hw * kDim + vrow]) ^ fold);
        sm.u1.fin.us[hw][m][vrow] =
            __ushort_as_half(__half_as_ushort(sm.ur[hw][4 + m][vrow]) ^ __half_as_ushort(sm.kr[hw][4 + m][vrow]));
        if (hw == 0) sm.khs[m][vrow] = sm.kr[0][8 + m][vrow];
      }
      if (tid < kHPC * kT) sm.Gm[tid / kT][tid % kT] = sm.G[tid / kT][tid % kT] + a_in;
      __syncthreads();
      store(sm, p, b, h0, lbase, l, tid);
      return;
    }
    const uint32_t nlive = kRank + static_cast<uint32_t>(l);
    // zero dead record rows (j >= 4 + l): their bytes are stale
    for (uint32_t i = tid; i < kHPC * kRows * 16; i += kThreads) {
      const uint32_t hh = i / (kRows * 16), j = (i / 16) % kRows, cc = i % 16;
      if (j >= nlive) {
        *reinterpret_cast<uint4*>(&sm.ur[hh][j][cc * 8]) = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(&sm.kr[hh][j][cc * 8]) = make_uint4(0, 0, 0, 0);
      }
    }
    if (tid < kHPC) {
      float G = l > 0 ? sm.G[tid][l - 1] : 0.f;
#pragma unroll
      for (uint32_t m = 0; m < kT; ++m) {
        G = G + sm.gate[tid][m];
        sm.Gm[tid][m] = G;
        sm.eGm[tid][m] = expf(G);
      }
    }
    __syncthreads();
    // weights e^{G_m - G_j}, base operand fp16(c e^{G_m} x^_m), small matrices
    {
      const uint32_t hh = tid / 128, k = tid % 128;
      const float ck = sm.c[hh][k];
#pragma unroll
      for (uint32_t m = 0; m < kT; ++m) {
        const float e = ck * sm.eGm[hh][m];
        sm.xs[hh][m][xcol(k)] = __float2half_rn(e * sm.kd[m][k]);
        sm.xs[hh][4 + m][xcol(k)] = __float2half_rn(e * sm.qd[m][k]);
      }
      if (k < kRows) {
        const uint32_t j = k;
#pragma unroll
        for (uint32_t m = 0; m < kT; ++m) {
          float wv = 0.f;
          if (j < kRank) {
            wv = sm.eGm[hh][m];
          } else if (j < nlive) {
            wv = exp_diff(sm.Gm[hh][m], sm.G[hh][j - kRank]);
          }
          sm.wrec[hh][j][m] = wv;
        }
      } else if (k < kRows + 1) {
        // P = (I + diag(beta) A)^{-1} diag(beta) and Qk with the decays, by forward substitution
        float Am[kT][kT], Pm[kT][kT];
#pragma unroll
        for (uint32_t m = 0; m < kT; ++m)
#pragma unroll
          for (uint32_t j = 0; j < kT; ++j) {
            const float d = j <= m ? exp_diff(sm.Gm[hh][m], sm.Gm[hh][j]) : 0.f;
            Am[m][j] = j < m ? sm.rawA[m][j] * d : 0.f;
            sm.Qk[hh][m][j] = j <= m ? sm.rawQ[m][j] * d : 0.f;
          }
#pragma unroll
        for (uint32_t m = 0; m < kT; ++m) {
          const float bm = sm.beta[hh][m];
#pragma unroll
          for (uint32_t j = 0; j < kT; ++j) {
            // row m of M^{-1} diag(beta): P[m][j] = beta_m (delta_mj beta_j/beta_m... ) solved as
            // P[m][:] = beta_m (e_m - sum_{i<m} A[m][i] P[i][:]) ... with diag(beta) on the right
            float v = (m == j ? 1.f : 0.f);
#pragma unroll
            for (uint32_t i = 0; i < m; ++i)
              v -= Am[m][i] * Pm[i][j];
            Pm[m][j] = bm * v;
          }
        }
        // Pm now = (I + diag(beta) A)^{-1} diag(beta) applied as U = Pm (V - W0)? see note below
#pragma unroll
        for (uint32_t m = 0; m < kT; ++m)
#pragma unroll
          for (uint32_t j = 0; j < kT; ++j)
            sm.Pm[hh][m][j] = Pm[m][j];
      }
    }
    __syncthreads();
    // ---- dots D[j][n] = <Kr_j, x^_n> on tensor cores: warp -> (head, m-tile of j, n-tile [exact|q hi] or [q lo])
    {
      const uint32_t hh = warp >> 2, mt = (warp >> 1) & 1, nt = warp & 1;
      uint32_t ka[2][16], qb[16];
#pragma unroll
      for (uint32_t hr = 0; hr < 2; ++hr) {
        const uint4* kp = reinterpret_cast<const uint4*>(&sm.kr[hh][16 * mt + g + 8 * hr][32 * t]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint4 v = kp[i];
          ka[hr][4 * i] = v.x, ka[hr][4 * i + 1] = v.y, ka[hr][4 * i + 2] = v.z, ka[hr][4 * i + 3] = v.w;
        }
      }
      {
        const uint4* xp = reinterpret_cast<const uint4*>(&sm.u1.xh[8 * nt + g][xcol(32 * t)]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint4 v = xp[i];
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
      const uint32_t j0 = 16 * mt + g;
      *reinterpret_cast<float2*>(&sm.u2.dots[hh][nt][j0][2 * t]) = make_float2(a0[0] + a1[0], a0[1] + a1[1]);
      *reinterpret_cast<float2*>(&sm.u2.dots[hh][nt][j0 + 8][2 * t]) = make_float2(a0[2] + a1[2], a0[3] + a1[3]);
    }
    __syncthreads();
    // coef[j][n] = D[j][n] w[j][m(n)] 2^13 -> hi / lo B operand rows
#pragma unroll
    for (uint32_t r = 0; r < 2; ++r) {
      const uint32_t i = tid + r * kThreads, hh = i / (kRows * 8), j = (i / 8) % kRows, n = i % 8;
      const float d =
          n < 4 ? sm.u2.dots[hh][0][j][n] : (sm.u2.dots[hh][0][j][n] + sm.u2.dots[hh][1][j][n]) * (1.f / 256.f);
      const float x = d * sm.wrec[hh][j][n & 3] * 8192.f;
      const __half hi = __float2half_rn(x);
      sm.bl[hh][n][j] = hi;
      sm.bl[hh][8 + n][j] = __float2half_rn(x - __half2float(hi));
    }
    __syncthreads();
    // ---- main products: warp -> head hw, rows 32 rbw .. + 32 (2 m-tiles); n = [W0 m 0..3 | O0 m 0..3]
    float val[2][4];
    {
      uint32_t xb[16];
      {
        const uint4* xp = reinterpret_cast<const uint4*>(&sm.xs[hw][g][xcol(32 * t)]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint4 v = xp[i];
          xb[4 * i] = v.x, xb[4 * i + 1] = v.y, xb[4 * i + 2] = v.z, xb[4 * i + 3] = v.w;
        }
      }
#pragma unroll
      for (uint32_t mt = 0; mt < 2; ++mt) {
        float az[4] = {0.f, 0.f, 0.f, 0.f}, ar[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (uint32_t s = 0; s < 8; ++s) {
          uint32_t a[4];
          i8x4_to_f16x2(zw[mt][0][s], a[0], a[2]);
          i8x4_to_f16x2(zw[mt][1][s], a[1], a[3]);
          mma_16816(az, a, xb[2 * s], xb[2 * s + 1]);
        }
#pragma unroll
        for (uint32_t q = 0; q < 2; ++q) {
          const uint32_t m = lane >> 3;
          uint32_t ua[4];
          ldsm_x4_trans(ua, &sm.ur[hw][16 * q + (lane & 7) + 8 * (m >> 1)][32 * rbw + 16 * mt + 8 * (m & 1)]);
          const uint32_t bh0 = *reinterpret_cast<const uint32_t*>(&sm.bl[hw][g][16 * q + 2 * t]);
          const uint32_t bh1 = *reinterpret_cast<const uint32_t*>(&sm.bl[hw][g][16 * q + 2 * t + 8]);
          const uint32_t bl0 = *reinterpret_cast<const uint32_t*>(&sm.bl[hw][8 + g][16 * q + 2 * t]);
          const uint32_t bl1 = *reinterpret_cast<const uint32_t*>(&sm.bl[hw][8 + g][16 * q + 2 * t + 8]);
          mma_16816(ar, ua, bh0, bh1);
          mma_16816(ar, ua, bl0, bl1);
        }
#pragma unroll
        for (int r = 0; r < 4; ++r)
          val[mt][r] = az[r] * srow[mt][r >> 1] + ar[r] * (1.f / 8192.f);
      }
    }
    // ---- per row: lane t of a quad takes row rho = t: (m-tile rho / 2, row g + 8 (rho % 2)); gather its 8 columns
    float rowv[8];
    {
      // exchange through smem: row lr = 16 mt + g + 8 hr holds columns 2t, 2t + 1 from lane (g, t)
#pragma unroll
      for (uint32_t mt = 0; mt < 2; ++mt) {
        *reinterpret_cast<float2*>(&sm.u1.fin.rowx[warp][16 * mt + g][2 * t]) = make_float2(val[mt][0], val[mt][1]);
        *reinterpret_cast<float2*>(&sm.u1.fin.rowx[warp][16 * mt + g + 8][2 * t]) = make_float2(val[mt][2], val[mt][3]);
      }
      __syncwarp();
      const uint32_t lr = 16 * (t >> 1) + g + 8 * (t & 1);
      const float4 r0 = *reinterpret_cast<const float4*>(&sm.u1.fin.rowx[warp][lr][0]);
      const float4 r1 = *reinterpret_cast<const float4*>(&sm.u1.fin.rowx[warp][lr][4]);
      rowv[0] = r0.x, rowv[1] = r0.y, rowv[2] = r0.z, rowv[3] = r0.w;
      rowv[4] = r1.x, rowv[5] = r1.y, rowv[6] = r1.z, rowv[7] = r1.w;
    }
    const uint32_t vrow = 32 * rbw + 16 * (t >> 1) + g + 8 * (t & 1);
    float o[kT];
    {
      float dv[kT], u[kT];
#pragma unroll
      for (uint32_t m = 0; m < kT; ++m)
        dv[m] = sm.vf[m][hw * kDim + vrow] - rowv[m];
#pragma unroll
      for (uint32_t m = 0; m < kT; ++m) {
        float a = 0.f;
#pragma unroll
        for (uint32_t j = 0; j <= m; ++j)
          a = fmaf(sm.Pm[hw][m][j], dv[j], a);
        u[m] = a;
      }
#pragma unroll
      for (uint32_t m = 0; m < kT; ++m) {
        float a = rowv[4 + m];
#pragma unroll
        for (uint32_t j = 0; j <= m; ++j)
          a = fmaf(sm.Qk[hw][m][j], u[j], a);
        o[m] = a;
        sm.u1.fin.us[hw][m][vrow] = __float2half_rn(u[m]);
      }
    }
    // ---- gated RMSNorm: sum of squares per (head, token) over the 128 rows
    {
      float ss[kT];
#pragma unroll
      for (uint32_t m = 0; m < kT; ++m) {
        ss[m] = o[m] * o[m];
#pragma unroll
        for (uint32_t of = 16; of >= 1; of >>= 1)
          ss[m] += __shfl_xor_sync(0xffffffffu, ss[m], of);
      }
      if (lane == 0) *reinterpret_cast<float4*>(sm.nsum[hw][rbw]) = make_float4(ss[0], ss[1], ss[2], ss[3]);
    }
    __syncthreads();
#pragma unroll
    for (uint32_t m = 0; m < kT; ++m) {
      const float tot = sm.nsum[hw][0][m] + sm.nsum[hw][1][m] + sm.nsum[hw][2][m] + sm.nsum[hw][3][m];
      const float rstd = rsqrtf(tot * (1.f / kDim) + p.eps);
      const float z = static_cast<float>(sm.xz[m][hw * kDim + vrow]);
      const float y = o[m] * rstd * sm.nw[vrow] * (z / (1.f + __expf(-z)));
      sm.u1.fin.outs[m][hw * kDim + vrow] = static_cast<bf16_t>(y);
    }
    __syncthreads();
    store(sm, p, b, h0, lbase, l, tid);
  }
};

template <uint32_t kC, bool kShell, uint32_t kHPC, uint32_t kMinBlocks>
__global__ void __launch_bounds__(128 * kHPC, kMinBlocks)
    gdn_lq_verify_kernel(const __grid_constant__ lqv::VerifyParams p) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  LQVerify<kC, kShell, kHPC>::run(*reinterpret_cast<typename LQVerify<kC, kShell, kHPC>::Smem*>(smem_raw), p);
}

template <uint32_t kC, bool kShell, uint32_t kHPC, uint32_t kMinBlocks>
void gdn_lq_verify(
    tvm::ffi::TensorView qkvz,
    tvm::ffi::TensorView ba,
    tvm::ffi::TensorView slot_idx,
    tvm::ffi::TensorView row_idx,
    tvm::ffi::TensorView pool,
    tvm::ffi::TensorView conv_state,
    tvm::ffi::TensorView conv_w,
    tvm::ffi::TensorView conv_win,
    tvm::ffi::TensorView A_log,
    tvm::ffi::TensorView dt_bias,
    tvm::ffi::TensorView norm_w,
    tvm::ffi::TensorView out,
    double eps) {
  using namespace host;
  using K = LQVerify<kC, kShell, kHPC>;
  auto NT = SymbolicSize{"tokens"}, N = SymbolicSize{"reqs"}, S = SymbolicSize{"slots"}, R = SymbolicSize{"rows"};
  auto CS = SymbolicSize{"conv_state_stride"}, WS = SymbolicSize{"conv_win_stride"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({NT, 12288}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(qkvz);
  TensorMatcher({NT, 64}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(ba);
  TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(slot_idx).verify(row_idx);
  TensorMatcher({-1, 32, 128, 128}).with_dtype<float>().with_device<kDLCUDA>(device).verify(pool);
  TensorMatcher({S, 8192, 3})
      .with_strides({CS, 3, 1})
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device)
      .verify(conv_state);
  TensorMatcher({8192, 4}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(conv_w);
  TensorMatcher({R, 8192, 6})
      .with_strides({WS, 6, 1})
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device)
      .verify(conv_win);
  TensorMatcher({32}).with_dtype<float>().with_device<kDLCUDA>(device).verify(A_log).verify(dt_bias);
  TensorMatcher({128}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(norm_w);
  TensorMatcher({NT, 4096}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(out);
  const uint32_t n = static_cast<uint32_t>(N.unwrap());
  CHECK_HOST(NT.unwrap() == static_cast<int64_t>(n) * lqv::kT) << "tokens must be N * T";
  if (n == 0) return;
  lqv::VerifyParams p{};
  p.qkvz = static_cast<const bf16_t*>(qkvz.data_ptr());
  p.ba = static_cast<const bf16_t*>(ba.data_ptr());
  p.slot_idx = static_cast<const int32_t*>(slot_idx.data_ptr());
  p.row_idx = static_cast<const int32_t*>(row_idx.data_ptr());
  p.pool = static_cast<uint8_t*>(pool.data_ptr());
  p.conv_state = static_cast<const bf16_t*>(conv_state.data_ptr());
  p.conv_w = static_cast<const bf16_t*>(conv_w.data_ptr());
  p.conv_win = static_cast<bf16_t*>(conv_win.data_ptr());
  p.A_log = static_cast<const float*>(A_log.data_ptr());
  p.dt_bias = static_cast<const float*>(dt_bias.data_ptr());
  p.norm_w = static_cast<const bf16_t*>(norm_w.data_ptr());
  p.out = static_cast<bf16_t*>(out.data_ptr());
  p.conv_state_stride = CS.unwrap();
  p.conv_win_stride = WS.unwrap();
  p.eps = static_cast<float>(eps);
  constexpr uint32_t kSmem = sizeof(typename K::Smem);
  const DLDevice dev = device.unwrap();
  runtime::set_max_dynamic_smem_per_device<gdn_lq_verify_kernel<kC, kShell, kHPC, kMinBlocks>>(dev, kSmem);
  LaunchKernel(n * (lqv::kHV / kHPC), K::kThreads, dev, kSmem)(gdn_lq_verify_kernel<kC, kShell, kHPC, kMinBlocks>, p);
}

}  // namespace sglang
