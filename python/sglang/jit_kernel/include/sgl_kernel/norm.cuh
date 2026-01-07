#pragma once
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace device {

namespace details {

__forceinline__ __device__ auto to_float2(nv_bfloat162 x) -> float2 {
  return __bfloat1622float2(x);
}

__forceinline__ __device__ auto to_float2(half2 x) -> float2 {
  return __half22float2(x);
}

template <typename T>
__forceinline__ __device__ auto from_float2(float2 x) -> T {
  if constexpr (std::is_same_v<T, nv_bfloat162>) {
    return __float22bfloat162_rn(x);
  } else if constexpr (std::is_same_v<T, half2>) {
    return __float22half2_rn(x);
  } else {
    static_assert(sizeof(T) == 0, "Unsupported type");
  }
}

inline constexpr auto resolve_norm_aligment(std::size_t byte_per_warp) -> std::size_t {
  if (byte_per_warp % kWarpThreads != 0) return 1;
  auto byte_per_lane = byte_per_warp / kWarpThreads;
  return (byte_per_lane % 16) == 0 ? 16  // at most 16B at a time for CUDA
         : byte_per_lane % 8 == 0  ? 8
         : byte_per_lane % 4 == 0  ? 4
                                   : 1;
}

}  // namespace details

template <int64_t kDim, uint32_t kNumThreads = kWarpThreads, typename PackedFloat, std::size_t N>
__forceinline__ __device__ aligned_vector<PackedFloat, N> apply_norm_impl(
    const aligned_vector<PackedFloat, N> input,
    const aligned_vector<PackedFloat, N> weight,
    const float eps,
    float* smem_buffer = nullptr) {
  static_assert(kNumThreads % kWarpThreads == 0);
  float sum_of_squares = 0.0f;

#pragma unroll
  for (auto i = 0u; i < N; ++i) {
    const auto fp32_input = details::to_float2(input[i]);
    sum_of_squares += fp32_input.x * fp32_input.x;
    sum_of_squares += fp32_input.y * fp32_input.y;
  }

  sum_of_squares = warp::reduce_sum(sum_of_squares);
  float norm_factor;
  if constexpr (kNumThreads > kWarpThreads) {
    // need to synchronize across the cta
    constexpr auto kNumWarps = kNumThreads / kWarpThreads;
    const auto warp_id = threadIdx.x / kWarpThreads;
    smem_buffer[warp_id] = sum_of_squares;
    __syncthreads();
    // use the first warp to reduce
    if (warp_id == 0) {
      const auto local_sum = (threadIdx.x < kNumWarps) ? smem_buffer[threadIdx.x] : 0.0f;
      sum_of_squares = warp::reduce_sum(local_sum);
      smem_buffer[0] = rsqrtf(sum_of_squares / kDim + eps);
    }
    __syncthreads();
    norm_factor = smem_buffer[0];
  } else {
    norm_factor = rsqrtf(sum_of_squares / kDim + eps);
  }

  aligned_vector<PackedFloat, N> output;

#pragma unroll
  for (auto i = 0u; i < N; ++i) {
    const auto fp32_input = details::to_float2(input[i]);
    const auto fp32_weight = details::to_float2(weight[i]);
    output[i] = details::from_float2<PackedFloat>({
        fp32_input.x * norm_factor * fp32_weight.x,
        fp32_input.y * norm_factor * fp32_weight.y,
    });
  }

  return output;
}

}  // namespace device
