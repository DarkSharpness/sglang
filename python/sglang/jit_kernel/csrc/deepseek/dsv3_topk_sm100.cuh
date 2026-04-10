/// \file deepseek/dsv3_topk_sm100.cuh
/// \brief Radix top-K selection kernel for SM100a (Blackwell).
///
/// Three algorithmic paths, dispatched per-row based on runtime seq_len:
///
///   Path A -- Filtered (seq_len <= FILTER_THRESH, single CTA):
///     FP16 coarse 8-bit histogram (1 global scan) -> suffix sum -> threshold.
///     Scatter + buffer threshold-bin indices (1 global scan, fused fine histogram).
///     Fine 4-round 8-bit radix from tiny smem buffer (~L/256 elements).
///     Matches AOT algorithm: 2 global scans + tiny refinement.
///
///   Path B -- Smem-cached (FILTER_THRESH < seq_len <= chunk_cap, single CTA):
///     Single global scan loads data to smem as ordered uint32.
///     All rounds operate from smem with prefix-mask filtering.
///
///   Path C -- Cooperative (seq_len > chunk_cap, multi-CTA cluster):
///     Same as Path B but with cluster histogram aggregation + merge.
///
/// Supports K = 512..2048.  NClusters = 1, 2, 4, 8.
#pragma once

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cooperative_groups.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace sglang {
namespace dsv3 {

namespace cg = cooperative_groups;
using device::AlignedVector;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr int TK_RADIX = 256;
static constexpr int TK_THREADS = 1024;
static constexpr int TK_MAX_K = 2048;
static constexpr int TK_ROUNDS = 4;         // fine radix: 4 bytes in uint32
static constexpr int TK_VEC = 4;            // float4 vectorized loads
static constexpr int TK_SMEM = 227 * 1024;  // SM100a max dynamic smem

// Filtered path: buffer for threshold-bin candidate indices (double-buffered).
// Sized same as AOT kernel. Overflow drops elements (negligible for FP16 key).
static constexpr int TK_BUF_N = 4096;

// Dispatch threshold: filtered path for short seq, smem-cached for longer.
static constexpr int TK_FILTER_THRESH = 32768;

// Smem-cached path: max data elements per CTA
static constexpr int TK_CACHE_OVER = TK_RADIX + TK_MAX_K + 4 + 8;
static constexpr int TK_MAX_CHUNK = TK_SMEM / 4 - TK_CACHE_OVER;

// ---------------------------------------------------------------------------
// Key conversions
// ---------------------------------------------------------------------------

/// Coarse key: float -> fp16 -> order-preserving -> top 8 bits.
/// Better 256-bin distribution than uint32>>24 (includes mantissa bits).
SGL_DEVICE uint8_t coarse_key(float x) {
  uint16_t h = __half_as_ushort(__float2half_rn(x));
  h = (h & 0x8000) ? static_cast<uint16_t>(~h) : static_cast<uint16_t>(h | 0x8000);
  return static_cast<uint8_t>(h >> 8);
}

/// Fine key: float -> full order-preserving uint32.
SGL_DEVICE uint32_t fine_key(float x) {
  uint32_t b = __float_as_uint(x);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

/// Extract r-th byte from a fine key (round 0 = MSB).
SGL_DEVICE uint8_t fine_byte(uint32_t fk, int r) {
  return static_cast<uint8_t>((fk >> (24 - r * 8)) & 0xFFu);
}

// ---------------------------------------------------------------------------
// Smem layouts (structs over extern __shared__)
// ---------------------------------------------------------------------------

/// Scalars shared across all paths.
struct TKScalars {
  int counter;       // result write position
  int threshold;     // current radix threshold bin
  int remaining_k;   // elements still needed
  int final_count;   // cluster-merge accumulator
  int warp_sums[8];  // suffix-sum workspace
  int buf_count[2];  // filtered path: double-buffered candidate counts
};

/// Filtered-path smem layout (no data cache, small footprint).
struct FilterSmem {
  int hist[TK_RADIX];
  int result[TK_MAX_K];
  TKScalars sc;
  int buf[2][TK_BUF_N];  // candidate index buffers

  static constexpr int SIZE_BYTES = (TK_RADIX + TK_MAX_K + sizeof(TKScalars) / 4 + 2 * TK_BUF_N) * 4;
};

/// Smem-cached path layout (data at front).
struct CacheSmem {
  // data[] occupies [0, chunk_cap) as uint32_t
  // followed by:
  int hist[TK_RADIX];
  int result[TK_MAX_K];
  TKScalars sc;

  /// Interpret raw smem as CacheSmem with data[] at offset 0.
  SGL_DEVICE static CacheSmem* from(int* s, int chunk_cap) {
    return reinterpret_cast<CacheSmem*>(s + chunk_cap);
  }
  SGL_DEVICE static uint32_t* data(int* s) {
    return reinterpret_cast<uint32_t*>(s);
  }
};

constexpr int topk_smem_bytes() {
  // max of both layouts
  constexpr int cache_bytes = (TK_MAX_CHUNK + TK_CACHE_OVER) * 4;
  constexpr int filter_bytes = FilterSmem::SIZE_BYTES;
  return cache_bytes > filter_bytes ? cache_bytes : filter_bytes;
}

// ---------------------------------------------------------------------------
// Warp-shuffle suffix sum  (4 barriers, all 1024 threads must call)
// Modifies hist[] in-place to suffix sums.  Writes sc->threshold, sc->remaining_k.
// ---------------------------------------------------------------------------

SGL_DEVICE void suffix_sum_find(int* hist, int rem_k, TKScalars* sc) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  int val = (tid < TK_RADIX) ? hist[tid] : 0;

#pragma unroll
  for (int i = 1; i < 32; i <<= 1) {
    int n = __shfl_down_sync(0xFFFFFFFF, val, i);
    if (lane + i < 32) val += n;
  }
  if (lane == 0 && warp < 8) sc->warp_sums[warp] = val;
  __syncthreads();

  if (tid < 8) {
    int w = sc->warp_sums[tid];
#pragma unroll
    for (int i = 1; i < 8; i <<= 1) {
      int n = __shfl_down_sync(0xFF, w, i);
      if (tid + i < 8) w += n;
    }
    sc->warp_sums[tid] = w;
  }
  __syncthreads();

  if (tid < TK_RADIX) {
    int excl = (warp < 7) ? sc->warp_sums[warp + 1] : 0;
    hist[tid] = val + excl;
  }
  __syncthreads();

  if (tid < TK_RADIX) {
    int s = hist[tid];
    int s_next = (tid < TK_RADIX - 1) ? hist[tid + 1] : 0;
    if (s >= rem_k && s_next < rem_k) {
      sc->threshold = tid;
      sc->remaining_k = rem_k - s_next;
    }
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------
// PATH A: Filtered radix (FP16 coarse + fine buffer, 2 global scans)
// ---------------------------------------------------------------------------

SGL_DEVICE void
filtered_topk(const float* __restrict__ input, int seq_len, int* __restrict__ output, int K, void* smem_raw) {
  auto& sm = *reinterpret_cast<FilterSmem*>(smem_raw);
  const int tid = threadIdx.x;

  // --- Init ---
  for (int i = tid; i < TK_RADIX; i += TK_THREADS)
    sm.hist[i] = 0;
  for (int i = tid; i < K; i += TK_THREADS)
    sm.result[i] = -1;
  if (tid == 0) {
    sm.sc.counter = 0;
    sm.sc.remaining_k = K;
    sm.sc.buf_count[0] = 0;
    sm.sc.buf_count[1] = 0;
  }
  __syncthreads();

  // --- Global scan 1: FP16 coarse histogram ---
  for (int i = tid; i < seq_len; i += TK_THREADS)
    atomicAdd(&sm.hist[coarse_key(input[i])], 1);
  __syncthreads();

  suffix_sum_find(sm.hist, K, &sm.sc);
  const int coarse_thr = sm.sc.threshold;

  // Early exit: all top-K strictly above threshold
  if (sm.sc.remaining_k == 0) {
    for (int i = tid; i < seq_len; i += TK_THREADS)
      if (coarse_key(input[i]) > coarse_thr) {
        int pos = atomicAdd(&sm.sc.counter, 1);
        if (pos < K) sm.result[pos] = i;
      }
    __syncthreads();
    goto write_out;
  }

  // --- Global scan 2: scatter + buffer + fused fine histogram ---
  for (int i = tid; i < TK_RADIX; i += TK_THREADS)
    sm.hist[i] = 0;
  __syncthreads();

  for (int i = tid; i < seq_len; i += TK_THREADS) {
    uint8_t ck = coarse_key(input[i]);
    if (ck > coarse_thr) {
      int pos = atomicAdd(&sm.sc.counter, 1);
      if (pos < K) sm.result[pos] = i;
    } else if (ck == coarse_thr) {
      int bi = atomicAdd(&sm.sc.buf_count[0], 1);
      if (bi < TK_BUF_N) sm.buf[0][bi] = i;
      // Fused: build fine histogram round 0 (byte 24..31 of fine key)
      atomicAdd(&sm.hist[fine_byte(fine_key(input[i]), 0)], 1);
    }
  }
  __syncthreads();

  // --- Fine passes from buffer ---
  {
    int cur = 0;  // current buffer phase

    for (int round = 0; round < TK_ROUNDS; round++) {
      int nxt = cur ^ 1;
      suffix_sum_find(sm.hist, sm.sc.remaining_k, &sm.sc);
      int fine_thr = sm.sc.threshold;
      bool is_last = (round == TK_ROUNDS - 1) || (sm.sc.remaining_k == 0);

      // Clear for next round
      if (!is_last)
        for (int i = tid; i < TK_RADIX; i += TK_THREADS)
          sm.hist[i] = 0;
      if (tid == 0) sm.sc.buf_count[nxt] = 0;
      __syncthreads();

      int nc = min(sm.sc.buf_count[cur], TK_BUF_N);
      for (int i = tid; i < nc; i += TK_THREADS) {
        int idx = sm.buf[cur][i];
        uint32_t fk = fine_key(input[idx]);  // re-read from global (L1 cached)
        uint8_t b = fine_byte(fk, round);

        if (b > fine_thr) {
          int pos = atomicAdd(&sm.sc.counter, 1);
          if (pos < K) sm.result[pos] = idx;
        } else if (b == fine_thr) {
          if (is_last) {
            int pos = atomicAdd(&sm.sc.counter, 1);
            if (pos < K) sm.result[pos] = idx;
          } else {
            int bi = atomicAdd(&sm.sc.buf_count[nxt], 1);
            if (bi < TK_BUF_N) sm.buf[nxt][bi] = idx;
            atomicAdd(&sm.hist[fine_byte(fk, round + 1)], 1);
          }
        }
      }
      __syncthreads();

      if (sm.sc.remaining_k == 0) break;
      cur = nxt;
    }
  }

write_out: {
  int cnt = min(sm.sc.counter, K);
  for (int i = tid; i < cnt; i += TK_THREADS)
    output[i] = sm.result[i];
  for (int i = cnt + tid; i < K; i += TK_THREADS)
    output[i] = -1;
}
}

// ---------------------------------------------------------------------------
// PATH B/C helpers: smem-cached radix
// ---------------------------------------------------------------------------

SGL_DEVICE void
load_and_hist(const float* __restrict__ src, uint32_t* __restrict__ dst, int* hist, int start, int count) {
  const int tid = threadIdx.x;
  const float* base = src + start;
  if ((reinterpret_cast<uintptr_t>(base) & 15) == 0) {
    const int vn = count / TK_VEC;
    for (int i = tid; i < vn; i += TK_THREADS) {
      AlignedVector<float, TK_VEC> v;
      v.load(base, i);
#pragma unroll
      for (int j = 0; j < TK_VEC; j++) {
        uint32_t fk = fine_key(v[j]);
        dst[i * TK_VEC + j] = fk;
        atomicAdd(&hist[fine_byte(fk, 0)], 1);
      }
    }
    for (int i = vn * TK_VEC + tid; i < count; i += TK_THREADS) {
      uint32_t fk = fine_key(base[i]);
      dst[i] = fk;
      atomicAdd(&hist[fine_byte(fk, 0)], 1);
    }
  } else {
    for (int i = tid; i < count; i += TK_THREADS) {
      uint32_t fk = fine_key(base[i]);
      dst[i] = fk;
      atomicAdd(&hist[fine_byte(fk, 0)], 1);
    }
  }
}

SGL_DEVICE void smem_scatter(
    const uint32_t* data,
    int count,
    int chunk_start,
    int round,
    int thr_byte,
    uint32_t prefix,
    uint32_t mask,
    int* result,
    TKScalars* sc,
    int K,
    int* next_hist,
    int next_round,
    bool build_next) {
  const int tid = threadIdx.x;
  for (int i = tid; i < count; i += TK_THREADS) {
    uint32_t fk = data[i];
    if ((fk & mask) != prefix) continue;
    uint8_t b = fine_byte(fk, round);
    if (b > thr_byte) {
      int pos = atomicAdd(&sc->counter, 1);
      if (pos < K) result[pos] = chunk_start + i;
    } else if (b == thr_byte && build_next) {
      atomicAdd(&next_hist[fine_byte(fk, next_round)], 1);
    }
  }
}

SGL_DEVICE void
smem_collect(const uint32_t* data, int count, int chunk_start, uint32_t pivot, int* result, TKScalars* sc, int K) {
  const int tid = threadIdx.x;
  for (int i = tid; i < count; i += TK_THREADS) {
    if (data[i] == pivot) {
      int pos = atomicAdd(&sc->counter, 1);
      if (pos < K) result[pos] = chunk_start + i;
    }
  }
}

SGL_DEVICE void
cached_topk(const float* __restrict__ input, int count, int* __restrict__ output, int K, int* smem_raw, int chunk_cap) {
  auto* sm = CacheSmem::from(smem_raw, chunk_cap);
  auto* data = CacheSmem::data(smem_raw);
  const int tid = threadIdx.x;

  for (int i = tid; i < TK_RADIX; i += TK_THREADS)
    sm->hist[i] = 0;
  for (int i = tid; i < K; i += TK_THREADS)
    sm->result[i] = -1;
  if (tid == 0) {
    sm->sc.counter = 0;
    sm->sc.remaining_k = K;
  }
  __syncthreads();

  load_and_hist(input, data, sm->hist, 0, count);
  __syncthreads();

  uint32_t prefix = 0, mask = 0;
  for (int round = 0; round < TK_ROUNDS; round++) {
    suffix_sum_find(sm->hist, sm->sc.remaining_k, &sm->sc);
    int thr = sm->sc.threshold;
    bool is_last = (round == TK_ROUNDS - 1) || (sm->sc.remaining_k == 0);

    if (!is_last)
      for (int i = tid; i < TK_RADIX; i += TK_THREADS)
        sm->hist[i] = 0;
    __syncthreads();

    smem_scatter(data, count, 0, round, thr, prefix, mask, sm->result, &sm->sc, K, sm->hist, round + 1, !is_last);
    __syncthreads();

    prefix |= ((uint32_t)thr << (24 - round * 8));
    mask |= (0xFFu << (24 - round * 8));
    if (sm->sc.remaining_k == 0) break;
    if (is_last) {
      smem_collect(data, count, 0, prefix, sm->result, &sm->sc, K);
      __syncthreads();
    }
  }

  int cnt = min(sm->sc.counter, K);
  for (int i = tid; i < cnt; i += TK_THREADS)
    output[i] = sm->result[i];
  for (int i = cnt + tid; i < K; i += TK_THREADS)
    output[i] = -1;
}

// ---------------------------------------------------------------------------
// Cluster histogram aggregation
// ---------------------------------------------------------------------------

template <int NClusters>
SGL_DEVICE void
aggregate_hist([[maybe_unused]] cg::cluster_group& cluster, int* local_hist, [[maybe_unused]] int block_rank) {
  if constexpr (NClusters == 1) {
    __syncthreads();
    return;
  }
  const int tid = threadIdx.x;
  cluster.sync();
  int sum = 0;
  if (tid < TK_RADIX)
    for (int r = 0; r < NClusters; r++)
      sum += cluster.map_shared_rank(local_hist, r)[tid];
  cluster.sync();
  if (tid < TK_RADIX) local_hist[tid] = sum;
  __syncthreads();
}

// ---------------------------------------------------------------------------
// PATH C: Cooperative multi-CTA smem-cached radix
// ---------------------------------------------------------------------------

template <int NClusters>
SGL_DEVICE void cooperative_topk(
    cg::cluster_group& cluster,
    int block_rank,
    const float* __restrict__ input,
    int seq_len,
    int* __restrict__ output,
    int K,
    int* smem_raw,
    int chunk_cap) {
  auto* sm = CacheSmem::from(smem_raw, chunk_cap);
  auto* data = CacheSmem::data(smem_raw);
  const int tid = threadIdx.x;

  int coop_chunk = (seq_len + NClusters - 1) / NClusters;
  coop_chunk = (coop_chunk + TK_VEC - 1) & ~(TK_VEC - 1);
  int my_start = block_rank * coop_chunk;
  int my_end = min(my_start + coop_chunk, seq_len);
  int my_count = max(0, my_end - my_start);

  for (int i = tid; i < TK_RADIX; i += TK_THREADS)
    sm->hist[i] = 0;
  for (int i = tid; i < K; i += TK_THREADS)
    sm->result[i] = -1;
  if (tid == 0) {
    sm->sc.counter = 0;
    sm->sc.remaining_k = K;
    sm->sc.final_count = 0;
  }
  __syncthreads();

  if (my_count > 0) load_and_hist(input, data, sm->hist, my_start, my_count);
  __syncthreads();
  aggregate_hist<NClusters>(cluster, sm->hist, block_rank);

  uint32_t prefix = 0, mask = 0;
  for (int round = 0; round < TK_ROUNDS; round++) {
    suffix_sum_find(sm->hist, sm->sc.remaining_k, &sm->sc);
    int thr = sm->sc.threshold;
    bool is_last = (round == TK_ROUNDS - 1) || (sm->sc.remaining_k == 0);

    if (!is_last)
      for (int i = tid; i < TK_RADIX; i += TK_THREADS)
        sm->hist[i] = 0;
    __syncthreads();

    smem_scatter(
        data, my_count, my_start, round, thr, prefix, mask, sm->result, &sm->sc, K, sm->hist, round + 1, !is_last);
    __syncthreads();

    prefix |= ((uint32_t)thr << (24 - round * 8));
    mask |= (0xFFu << (24 - round * 8));
    if (!is_last) aggregate_hist<NClusters>(cluster, sm->hist, block_rank);
    if (sm->sc.remaining_k == 0) break;
    if (is_last) {
      smem_collect(data, my_count, my_start, prefix, sm->result, &sm->sc, K);
      __syncthreads();
    }
  }

  // --- Merge across cluster ---
  int local_count = min(sm->sc.counter, K);
  int topk_start;
  if constexpr (NClusters > 1) {
    if (block_rank == 0 && tid == 0) sm->sc.final_count = local_count;
    cluster.sync();
    if (block_rank > 0 && tid == 0)
      topk_start = atomicAdd(cluster.map_shared_rank(&sm->sc.final_count, 0), local_count);
    cluster.sync();
    if (block_rank == 0) topk_start = 0;
    if (tid == 0) sm->sc.threshold = topk_start;
    __syncthreads();
    topk_start = sm->sc.threshold;
  } else {
    topk_start = 0;
  }

  for (int i = tid; i < local_count; i += TK_THREADS) {
    int gpos = topk_start + i;
    if (gpos < K) output[gpos] = sm->result[i];
  }
  if (block_rank == 0) {
    int total;
    if constexpr (NClusters > 1)
      total = min(*cluster.map_shared_rank(&sm->sc.final_count, 0), K);
    else
      total = min(local_count, K);
    for (int i = total + tid; i < K; i += TK_THREADS)
      output[i] = -1;
  }
}

// ---------------------------------------------------------------------------
// Entry kernel: dispatch only
// ---------------------------------------------------------------------------

template <int NClusters>
__global__ void __cluster_dims__(NClusters, 1, 1) __launch_bounds__(1024) dsv3_topk_kernel(
    const float* __restrict__ logits,
    const int* __restrict__ seq_lens,
    int* __restrict__ out_indices,
    int K,
    int logits_stride,
    int out_stride,
    int chunk_cap,
    int batch_size) {
  extern __shared__ int topk_smem[];

  [[maybe_unused]] cg::cluster_group cluster = cg::this_cluster();
  const int block_rank = NClusters > 1 ? (int)cluster.block_rank() : 0;
  const int batch_idx = (int)blockIdx.x / NClusters;
  if (batch_idx >= batch_size) return;

  const int tid = threadIdx.x;
  const int seq_len = seq_lens[batch_idx];
  int* output = out_indices + (int64_t)batch_idx * out_stride;
  const float* input = logits + (int64_t)batch_idx * logits_stride;

  // --- Trivial ---
  if (seq_len <= K) {
    if (block_rank == 0) {
      for (int i = tid; i < seq_len; i += TK_THREADS)
        output[i] = i;
      for (int i = seq_len + tid; i < K; i += TK_THREADS)
        output[i] = -1;
    }
    return;
  }

  // --- Dispatch ---
  if (seq_len <= TK_FILTER_THRESH) {
    if (block_rank != 0) return;
    filtered_topk(input, seq_len, output, K, topk_smem);
  } else if (NClusters == 1 || seq_len <= chunk_cap) {
    if (block_rank != 0) return;
    cached_topk(input, min(seq_len, chunk_cap), output, K, topk_smem, chunk_cap);
  } else {
    cooperative_topk<NClusters>(cluster, block_rank, input, seq_len, output, K, topk_smem, chunk_cap);
  }
}

// ---------------------------------------------------------------------------
// Host launch
// ---------------------------------------------------------------------------

void launch_dsv3_topk(
    const float* logits,
    const int* seq_lens,
    int* out_indices,
    int* /*overflow_buf*/,
    int K,
    int logits_stride,
    int out_stride,
    int /*ov_stride*/,
    int batch_size,
    int num_clusters,
    cudaStream_t stream) {
  const int sb = topk_smem_bytes();
  const int chunk_cap = TK_MAX_CHUNK;

  const auto kernel = [&] {
    switch (num_clusters) {
      case 2:
        return dsv3_topk_kernel<2>;
      case 4:
        return dsv3_topk_kernel<4>;
      case 8:
        return dsv3_topk_kernel<8>;
      default:
        return dsv3_topk_kernel<1>;
    }
  }();
  cudaFuncSetAttribute(reinterpret_cast<void*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, sb);

  void* args[] = {
      const_cast<void*>(reinterpret_cast<const void*>(&logits)),
      const_cast<void*>(reinterpret_cast<const void*>(&seq_lens)),
      reinterpret_cast<void*>(&out_indices),
      reinterpret_cast<void*>(&K),
      reinterpret_cast<void*>(&logits_stride),
      reinterpret_cast<void*>(&out_stride),
      const_cast<void*>(reinterpret_cast<const void*>(&chunk_cap)),
      reinterpret_cast<void*>(&batch_size),
  };

  cudaLaunchConfig_t config = {};
  config.blockDim.x = TK_THREADS;
  config.gridDim.x = batch_size * num_clusters;
  config.dynamicSmemBytes = sb;
  config.stream = stream;

  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {(unsigned)num_clusters, 1, 1};
  config.attrs = attrs;
  config.numAttrs = 1;

  cudaLaunchKernelExC(&config, reinterpret_cast<void*>(kernel), args);
}

}  // namespace dsv3
}  // namespace sglang
