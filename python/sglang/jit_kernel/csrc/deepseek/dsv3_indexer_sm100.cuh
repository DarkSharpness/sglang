/// \file deepseek/dsv3_indexer_sm100_v2.cuh
/// \brief V2 fused paged-logits kernel for DeepSeek V3 sparse indexer.
///
/// Key differences from v1 (dsv3_indexer_sm100.cuh):
///   - 2D scheduler (q_begin,k_begin,q_end,k_end) instead of fixed SM mapping
///   - Separate Q loading warp with multi-stage Q pipeline
///   - Merged K data + K scales loading (single ldg warp per group)
///   - Per-head Q scale support
///
/// Computes:
///   logits[b, t] = sum_h( relu(q_fp8[b,h] . K_fp8[b,t,h]) * q_scale[b,h] ) * k_scale[t]
///
/// Requires SM100a (Blackwell).  Always uses PDL.
///
/// Warp organization (EPILOGUE_WARPGRPS=2, 13 warps total):
///   Warps 0-7:   Epilogue (2 groups x 4 warps) -- read TMEM, ReLU, weight, write
///   Warps 8-9:   LDG K   (1 per group)         -- TMA load K data + K scales
///   Warps 10-11: MMA     (1 per group)          -- issue tcgen05.mma
///   Warp 12:     LDG Q   (shared)               -- TMA load Q data + Q scales

#pragma once

#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/deepseek/sm100a_utils.cuh>

#include <cassert>
#include <cstdint>

namespace sglang::dsv3 {

using device::div_ceil;

// ---------------------------------------------------------------------------
// TMA tensor-map setup (host-side, same layout as v1)
// ---------------------------------------------------------------------------

template <int Q_NEXT>
inline void
prep_tmaps(CUtensorMap* q_tmap, CUtensorMap* k_tmap, uint8_t* q_ptr, uint8_t* k_ptr, int batch_size, int num_pages) {
  using namespace sglang::dsv3;
  uint32_t q_boxDim[3] = {128, 64 * Q_NEXT, 1};
  uint32_t k_boxDim[3] = {128, 64, 1};
  uint64_t q_globalDim[3] = {128, 64 * Q_NEXT, (uint64_t)batch_size};
  uint64_t k_globalDim[3] = {128, 64, (uint64_t)num_pages};
  uint64_t q_globalStrides[2] = {128, 128 * 64 * Q_NEXT};
  uint64_t k_globalStrides[2] = {128, 132 * 64};
  init_tensormap_nd<3>(q_tmap, q_ptr, q_globalDim, q_globalStrides, q_boxDim);
  init_tensormap_nd<3>(k_tmap, k_ptr, k_globalDim, k_globalStrides, k_boxDim);
}

// ---------------------------------------------------------------------------
// Schedule metadata (per-SM work range in the 2D (batch, kv) space)
// ---------------------------------------------------------------------------

struct alignas(16) ScheduleMetadata {
  uint32_t q_begin;
  uint32_t k_begin;
  uint32_t q_end;
  uint32_t k_end;
};

// ---------------------------------------------------------------------------
// Kernel parameters
// ---------------------------------------------------------------------------

struct IndexerParams {
  const float* __restrict__ k_s;                  // K per-token scales (see K_SCALE_STRIDE)
  const float* __restrict__ q_s;                  // Q per-head scales/weights: [batch, 64]
  const int32_t* __restrict__ page_table;         // [batch, page_table_stride]
  const int32_t* __restrict__ seq_lens;           // [batch]
  float* __restrict__ out_logits;                 // [batch, logits_stride]
  const ScheduleMetadata* __restrict__ metadata;  // [num_sms]
  int64_t page_table_stride;
  int64_t logits_stride;
};

// ---------------------------------------------------------------------------
// Work scheduler -- iterates (q_idx, k_idx) from (q_begin,k_begin) to
// (q_end,k_end) with a given step in the k dimension.
// ---------------------------------------------------------------------------

struct Scheduler {
  SGL_DEVICE Scheduler(ScheduleMetadata meta, const int32_t* seq_lens, uint32_t step)
      : q_idx(meta.q_begin),
        k_idx(meta.k_begin),
        seq_lens(seq_lens),
        q_end_idx(meta.q_end),
        k_end_idx(meta.k_end),
        step(step) {
    update_cached();
  }

  /// Advance by one step.  Returns true when q_idx changed.
  SGL_DEVICE bool advance() {
    k_idx += step;
    if (k_idx >= cached_seq_len) {
      q_idx += 1;
      k_idx = 0;
      update_cached();
      return true;
    }
    return false;
  }

  SGL_DEVICE bool is_valid() const {
    if (q_idx < q_end_idx) return true;
    return q_idx == q_end_idx && k_idx < k_end_idx;
  }

  uint32_t q_idx;
  uint32_t k_idx;
  const int32_t* __restrict__ const seq_lens;
  const uint32_t q_end_idx;
  const uint32_t k_end_idx;
  const uint32_t step;
  uint32_t cached_seq_len;

 private:
  SGL_DEVICE void update_cached() {
    if (q_idx <= q_end_idx) cached_seq_len = seq_lens[q_idx];
  }
};

// ===========================================================================
// KernelTrait -- all constants, smem layout, and device-side kernel logic
// ===========================================================================

SGL_DEVICE auto get_phase(uint32_t iter, uint32_t stages) -> uint2 {
  return make_uint2(iter % stages, (iter / stages) & 1);
}

struct KernelTrait {
  // ----- Tile / pipeline sizes -----
  static constexpr int Q_NEXT = 1;  // only support 1 now
  static constexpr int EPILOGUE_WARPGRPS = 2;
  static constexpr int Q_LDG_STAGES = 2;
  static constexpr int K_LDG_STAGES = 4;
  static constexpr int K_MMA_STAGES = 4;

  // ----- Dimension constants -----
  static constexpr uint32_t NUM_HEADS = 64;
  static constexpr uint32_t HEAD_DIM = 128;
  static constexpr uint32_t PAGE_SIZE = 64;
  static constexpr uint32_t BLOCK_KV = 2 * PAGE_SIZE;                   // 128 tokens per group per iter
  static constexpr uint32_t PAGES_PER_GROUP = BLOCK_KV / PAGE_SIZE;     // 2
  static constexpr uint32_t VEC_LD = 16;                                // vectorised TMEM load width
  static constexpr uint32_t SCHED_STEP = BLOCK_KV * EPILOGUE_WARPGRPS;  // 256 tokens per scheduler step

  // ----- Thread / warp layout -----
  static constexpr uint32_t NUM_WARPS = EPILOGUE_WARPGRPS * 6 + 1;                     // 13
  static constexpr uint32_t THREADS = NUM_WARPS * 32;                                  // 416
  static constexpr uint32_t TMEM_COLS = NUM_HEADS * K_MMA_STAGES * EPILOGUE_WARPGRPS;  // 512

  static constexpr uint32_t LDG_K_OFF = EPILOGUE_WARPGRPS * 4;  // 8
  static constexpr uint32_t MMA_OFF = EPILOGUE_WARPGRPS * 5;    // 10
  static constexpr uint32_t LDG_Q_OFF = EPILOGUE_WARPGRPS * 6;  // 12

  static constexpr uint32_t EPI_THREADS_PER_GRP = 4 * 32;                               // 128
  static constexpr uint32_t EPI_THREADS_ALL = EPILOGUE_WARPGRPS * EPI_THREADS_PER_GRP;  // 256

  // ----- K scale stride (floats) between pages -----
  // K layout per page: PAGE_SIZE * (HEAD_DIM + 4) bytes; k_scale points to scales.
  static constexpr uint32_t K_SCALE_STRIDE = PAGE_SIZE * (HEAD_DIM + sizeof(float)) / sizeof(float);  // 2112

  // ----- MMA instruction descriptor:  f8, N=NUM_HEADS, K=HEAD_DIM -----
  static constexpr uint32_t IDESC = (1U << 4U) | ((NUM_HEADS >> 3U) << 17U) | ((HEAD_DIM >> 4U) << 24U);

  static_assert(THREADS <= 1024 && THREADS % 32 == 0);
  static_assert(TMEM_COLS <= 512);

  // ----- Shared-memory layout -----
  struct SMemLayout {
    float q_s[Q_LDG_STAGES][NUM_HEADS];
    float k_s[EPILOGUE_WARPGRPS][K_LDG_STAGES][BLOCK_KV];                           // 128 per stage
    alignas(1024) uint8_t q[Q_LDG_STAGES][NUM_HEADS][HEAD_DIM];                     // 8192 per stage
    alignas(1024) uint8_t k[EPILOGUE_WARPGRPS][K_LDG_STAGES][BLOCK_KV * HEAD_DIM];  // 16384 per stage
  };

  // ----- TMA byte counts (per page) -----
  static constexpr uint32_t K_DATA_1P = PAGE_SIZE * HEAD_DIM;        // 8192
  static constexpr uint32_t K_SCALE_1P = PAGE_SIZE * sizeof(float);  // 256

  // =====================================================================
  // Device kernel
  // =====================================================================

  SGL_DEVICE static void kernel(const CUtensorMap* q_tmap, const CUtensorMap* k_tmap, const IndexerParams& params) {
    using namespace sglang::dsv3;
    const auto metadata = params.metadata[blockIdx.x];

    // ----- Barriers -----
    static __shared__ int32_t s_tmem_addr;
    static __shared__ uint64_t q_ready[Q_LDG_STAGES];
    static __shared__ uint64_t q_empty[Q_LDG_STAGES];
    static __shared__ uint64_t k_ready[EPILOGUE_WARPGRPS][K_LDG_STAGES];
    static __shared__ uint64_t k_empty[EPILOGUE_WARPGRPS][K_LDG_STAGES];
    static __shared__ uint64_t mma_ready[EPILOGUE_WARPGRPS][K_MMA_STAGES];
    static __shared__ uint64_t mma_empty[EPILOGUE_WARPGRPS][K_MMA_STAGES];

    alignas(1024) extern __shared__ SMemLayout smem[];

    // ----- Warp identification -----
    const auto warp_id = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const auto lane_id = threadIdx.x % 32;
    const auto warp_grp_id = warp_id / 4;

    const bool is_epilogue = warp_id < LDG_K_OFF;
    const bool is_ldg_k = warp_id >= LDG_K_OFF && warp_id < MMA_OFF;
    const bool is_mma = warp_id >= MMA_OFF && warp_id < LDG_Q_OFF;
    const bool is_ldg_q = warp_id >= LDG_Q_OFF;

    const auto k_group = [&] {
      if (is_epilogue) return warp_grp_id;
      if (is_ldg_k) return warp_id - LDG_K_OFF;
      if (is_mma) return warp_id - MMA_OFF;
      return 0u;
    }();

    // ----- MMA descriptor helper -----
    const auto make_desc_a = [](int addr) -> uint64_t {
      constexpr int SBO = HEAD_DIM * 8;
      return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
    };

    // =================================================================
    // Initialisation
    // =================================================================
    if (warp_id == 0) {
      tcgen05_alloc(TMEM_COLS, &s_tmem_addr);
    } else if (warp_id == 1) {
      if (lane_id < Q_LDG_STAGES) {
        mbarrier_init(&q_ready[lane_id], 1);
        mbarrier_init(&q_empty[lane_id], EPI_THREADS_ALL);
      }
    } else if (warp_id == 2) {
      if (lane_id < EPILOGUE_WARPGRPS * K_LDG_STAGES) {
        const auto g = lane_id / K_LDG_STAGES;
        const auto s = lane_id % K_LDG_STAGES;
        mbarrier_init(&k_ready[g][s], 1);
        mbarrier_init(&k_empty[g][s], EPI_THREADS_PER_GRP);
      }
    } else if (warp_id == 3) {
      if (lane_id < EPILOGUE_WARPGRPS * K_MMA_STAGES) {
        const auto g = lane_id / K_MMA_STAGES;
        const auto s = lane_id % K_MMA_STAGES;
        mbarrier_init(&mma_ready[g][s], 1);
        mbarrier_init(&mma_empty[g][s], EPI_THREADS_PER_GRP);
      }
    }

    device::PDLWaitPrimary<true>();

    __syncthreads();
    const int taddr = s_tmem_addr;
    auto sched = Scheduler{metadata, params.seq_lens, SCHED_STEP};

    // =================================================================
    // LDG K warp -- load 2 pages of K FP8 data + scales per iteration
    // =================================================================
    if (is_ldg_k) {
      if (elect_sync()) {
        for (uint32_t iter = 0; sched.is_valid(); ++iter, sched.advance()) {
          const auto [stage, phase] = get_phase(iter, K_LDG_STAGES);
          if (iter >= K_LDG_STAGES) {
            mbarrier_wait(&k_empty[k_group][stage], phase ^ 1);
          }

          const auto pg_row = sched.q_idx * params.page_table_stride;
          const int base_pg = sched.k_idx / PAGE_SIZE + k_group * PAGES_PER_GROUP;
          const int last_pg = div_ceil((int)sched.cached_seq_len, (int)PAGE_SIZE);
          const auto bar = &k_ready[k_group][stage];

          uint32_t tx = 0;
          // Page 0
          if (base_pg < last_pg) {
            const auto p0 = params.page_table[pg_row + base_pg];
            tma_3d_gmem2smem(smem->k[k_group][stage], k_tmap, 0, 0, p0, bar);
            tma_1d_gmem2smem(smem->k_s[k_group][stage], &params.k_s[p0 * K_SCALE_STRIDE], K_SCALE_1P, bar);
            tx += K_DATA_1P + K_SCALE_1P;
          }
          // Page 1
          if (base_pg + 1 < last_pg) {
            const auto p1 = params.page_table[pg_row + base_pg + 1];
            tma_3d_gmem2smem(smem->k[k_group][stage] + PAGE_SIZE * HEAD_DIM, k_tmap, 0, 0, p1, bar);
            tma_1d_gmem2smem(smem->k_s[k_group][stage] + PAGE_SIZE, &params.k_s[p1 * K_SCALE_STRIDE], K_SCALE_1P, bar);
            tx += K_DATA_1P + K_SCALE_1P;
          }
          mbarrier_arrive_expect_tx(bar, tx);
        }
      }
    }
    // =================================================================
    // LDG Q warp -- load Q FP8 data + per-head scales on batch change
    // =================================================================
    else if (is_ldg_q) {
      if (elect_sync()) {
        constexpr uint32_t Q_DATA_B = NUM_HEADS * HEAD_DIM;
        constexpr uint32_t Q_SCALE_B = NUM_HEADS * sizeof(float);

        for (uint32_t iter = 0, q_idx = sched.q_idx; q_idx <= sched.q_end_idx; ++iter, ++q_idx) {
          const auto stg = iter % Q_LDG_STAGES;
          if (iter >= Q_LDG_STAGES) {
            const auto phase = ((iter / Q_LDG_STAGES) & 1);
            mbarrier_wait(&q_empty[stg], phase ^ 1);
          }
          const auto bar = &q_ready[stg];
          tma_3d_gmem2smem(smem->q[stg], q_tmap, 0, 0, q_idx, bar);
          tma_1d_gmem2smem(smem->q_s[stg], &params.q_s[q_idx * NUM_HEADS], Q_SCALE_B, bar);
          mbarrier_arrive_expect_tx(bar, Q_DATA_B + Q_SCALE_B);
        }
      }
    }
    // =================================================================
    // MMA warp -- issue tcgen05 FP8 MMA:  K[128x128] x Q[128x64] -> TMEM
    // =================================================================
    else if (is_mma) {
      if (elect_sync()) {
        const int grp_taddr = taddr + k_group * K_MMA_STAGES * NUM_HEADS;
        uint32_t q_iter = 0, q_stg = 0, cur_q = UINT32_MAX;
        for (uint32_t iter = 0; sched.is_valid(); ++iter, sched.advance()) {
          if (cur_q != sched.q_idx) {
            cur_q = sched.q_idx;
            const auto [qs, qp] = get_phase(q_iter++, Q_LDG_STAGES);
            q_stg = qs;
            mbarrier_wait(&q_ready[qs], qp);
          }

          const auto [k_stage, k_phase] = get_phase(iter, K_LDG_STAGES);
          mbarrier_wait(&k_ready[k_group][k_stage], k_phase);
          const auto [m_stage, m_phase] = get_phase(iter, K_MMA_STAGES);
          if (iter >= K_MMA_STAGES) {
            mbarrier_wait(&mma_empty[k_group][m_stage], m_phase ^ 1);
          }

          asm volatile("tcgen05.fence::after_thread_sync;");
          const int cur_t = grp_taddr + m_stage * NUM_HEADS;
          const int ka = cvt_addr(smem->k[k_group][k_stage]);
          const int qa = cvt_addr(smem->q[q_stg]);
          for (int kk = 0; kk < (int)HEAD_DIM / 32; kk++) {
            tcgen05_mma_f8(cur_t, make_desc_a(ka + 32 * kk), make_desc_a(qa + 32 * kk), IDESC, kk > 0);
          }

          tcgen05_mma_arrive(&mma_ready[k_group][m_stage]);
        }
      }
    }
    // =================================================================
    // Epilogue warps -- read TMEM, apply ReLU x q_scale, write logits
    // =================================================================
    // =================================================================
    else if (is_epilogue) {
      const int grp_taddr = taddr + k_group * K_MMA_STAGES * NUM_HEADS;
      const int row_off = (warp_id % 4) * 32;
      const int local_tok = lane_id + row_off;
      float qs_reg[NUM_HEADS];

      uint32_t q_iter = 0, cur_q = UINT32_MAX;
      for (uint32_t iter = 0; sched.is_valid(); ++iter, sched.advance()) {
        // ---- Q pipeline ----
        if (sched.q_idx != cur_q) {
          if (q_iter > 0) {
            const int old = (q_iter - 1) % Q_LDG_STAGES;
            mbarrier_arrive(&q_empty[old]);
          }
          const auto [q_stage, q_phase] = get_phase(q_iter++, Q_LDG_STAGES);
          mbarrier_wait(&q_ready[q_stage], q_phase);
          for (int h = 0; h < (int)NUM_HEADS; h++) {
            qs_reg[h] = smem->q_s[q_stage][h];
          }
          cur_q = sched.q_idx;
        }

        const auto [k_stage, k_phase] = get_phase(iter, K_LDG_STAGES);
        const auto [m_stage, m_phase] = get_phase(iter, K_MMA_STAGES);

        // ---- Wait for K data + scales ----
        mbarrier_wait(&k_ready[k_group][k_stage], k_phase);
        const float k_sc = smem->k_s[k_group][k_stage][local_tok];

        // ---- Wait for MMA result ----
        mbarrier_wait(&mma_ready[k_group][m_stage], m_phase);
        asm volatile("tcgen05.fence::after_thread_sync;");

        // ---- Weighted ReLU sum over heads ----
        const int cur_t = grp_taddr + m_stage * NUM_HEADS;
        constexpr int ACC_N = 2;
        float2 acc[ACC_N] = {make_float2(0, 0), make_float2(0, 0)};

        for (int off = 0; off < (int)NUM_HEADS; off += (int)VEC_LD) {
          float tmp[VEC_LD];
          tcgen05_ld_32x32b<VEC_LD>(cur_t + (row_off << 16) + off, tmp);
          for (int v = 0; v < (int)VEC_LD; v += 2) {
            float2 relu = make_float2(fmaxf(0.0f, tmp[v]), fmaxf(0.0f, tmp[v + 1]));
            float2 qs = make_float2(qs_reg[off + v], qs_reg[off + v + 1]);
            acc[(v / 2) % ACC_N] = __ffma2_rn(relu, qs, acc[(v / 2) % ACC_N]);
          }
        }

        float sum = 0.0f;
        for (int v = 0; v < ACC_N; v++)
          sum += acc[v].x + acc[v].y;

        // ---- Write logit ----
        const int token_pos = sched.k_idx + k_group * (int)BLOCK_KV + local_tok;
        if (token_pos < (int)sched.cached_seq_len) {
          params.out_logits[sched.q_idx * params.logits_stride + token_pos] = sum * k_sc;
        }

        // ---- Signal completion ----
        mbarrier_arrive(&mma_empty[k_group][m_stage]);
        mbarrier_arrive(&k_empty[k_group][k_stage]);
      }

      // Release final Q
      if (q_iter > 0) {
        const int old = (q_iter - 1) % Q_LDG_STAGES;
        mbarrier_arrive(&q_empty[old]);
      }
    }

    // =================================================================
    // Cleanup
    // =================================================================
    device::PDLTriggerSecondary<true>();
    __syncthreads();
    if (warp_id == 0) tcgen05_dealloc(TMEM_COLS, taddr);
  }
};

// ===========================================================================
// Global kernel entry point
// ===========================================================================

__global__ __launch_bounds__(KernelTrait::THREADS, 1) void dsv3_indexer_kernel(
    const __grid_constant__ CUtensorMap q_tmap,
    const __grid_constant__ CUtensorMap k_tmap,
    const IndexerParams params) {
  using namespace sglang::dsv3;

  if (threadIdx.x == 0) {
    prefetch_tma_descriptor(&q_tmap);
    prefetch_tma_descriptor(&k_tmap);
  }

  KernelTrait::kernel(&q_tmap, &k_tmap, params);
}

// ===========================================================================
// Host-side launch wrapper
// ===========================================================================

inline void launch_dsv3_indexer(
    uint8_t* q_ptr,
    uint8_t* k_ptr,
    float* q_scale,
    int* seq_lens,
    int* page_table,
    float* logits,
    ScheduleMetadata* metadata,
    int num_pages,
    int batch_size,
    int num_sms,
    int page_table_stride,
    int logits_stride,
    cudaStream_t stream) {
  CUtensorMap q_tmap, k_tmap;
  prep_tmaps<KernelTrait::Q_NEXT>(&q_tmap, &k_tmap, q_ptr, k_ptr, batch_size, num_pages);

  constexpr size_t smem_bytes = sizeof(KernelTrait::SMemLayout) + 1024;
  sglang::dsv3::setup_kernel_smem_once<dsv3_indexer_kernel, smem_bytes>();

  const auto params = IndexerParams{
      .k_s = reinterpret_cast<const float*>(k_ptr + KernelTrait::PAGE_SIZE * KernelTrait::HEAD_DIM),
      .q_s = q_scale,
      .page_table = page_table,
      .seq_lens = seq_lens,
      .out_logits = logits,
      .metadata = metadata,
      .page_table_stride = page_table_stride,
      .logits_stride = logits_stride,
  };
  host::LaunchKernel(num_sms, KernelTrait::THREADS, stream, smem_bytes)
      .enable_pdl(true)(dsv3_indexer_kernel, q_tmap, k_tmap, params);
}

}  // namespace sglang::dsv3
