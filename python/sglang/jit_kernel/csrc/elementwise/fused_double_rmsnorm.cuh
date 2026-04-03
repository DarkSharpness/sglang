#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

namespace {

struct FusedDoubleRMSNormParams {
  const void* input;
  const void* __restrict__ weight1;
  const void* __restrict__ weight2;
  const void* residual_in;
  void* output;
  void* residual_out;
  uint32_t num_tokens;
  float eps;
};

// Fused double RMSNorm kernel (CTA-level, 16B vector per thread):
//   normed1      = rmsnorm(input, weight1, eps)
//   residual_out = normed1 + residual_in
//   output       = rmsnorm(residual_out, weight2, eps)
//
// All tensors are assumed contiguous with stride == kDim.
//
// Register-pressure strategy:
//   - Each thread loads one 16B vector (4 regs) per tensor.
//   - weight2 and residual are loaded early so their memory latency
//     overlaps with the first norm's reduction, but are consumed only
//     in later phases once input/weight1 are dead.
//   - No "doubling" (two vectors per thread) to stay within 32 regs.
template <int64_t kDim, bool kUsePDL, typename Float>
__global__ void fused_double_rmsnorm_cta(const FusedDoubleRMSNormParams __grid_constant__ params) {
  using namespace device;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 4>;  // 16B per thread

  constexpr uint32_t kNumThreads = (kDim / 256) * kWarpThreads;
  constexpr uint32_t kNumWarps = kNumThreads / kWarpThreads;

  const auto& [input, weight1_ptr, weight2_ptr, residual_in, output, residual_out, num_tokens, eps] = params;
  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  __shared__ float smem[33];

  PDLWaitPrimary<kUsePDL>();

  const auto row = static_cast<int64_t>(blockIdx.x) * kDim;
  const auto in_ptr = pointer::offset<Float>(input, row);
  const auto res_in_ptr = pointer::offset<Float>(residual_in, row);

  // Load input and weight1
  const auto input_vec = gmem.load(in_ptr);
  const auto w1_vec = gmem.load(weight1_ptr);

  // Early-load residual and weight2 ??? their memory latency overlaps
  // with the first RMSNorm reduction below.
  const auto res_vec = gmem.load(res_in_ptr);
  const auto w2_vec = gmem.load(weight2_ptr);

  // --- First RMSNorm: compute sum-of-squares of input ---
  float sum1 = 0.0f;
#pragma unroll
  for (auto j = 0u; j < 4u; ++j) {
    const auto [x, y] = cast<fp32x2_t>(input_vec[j]);
    sum1 += x * x + y * y;
  }

  // CTA reduce
  sum1 = warp::reduce_sum(sum1);
  const auto warp_id = threadIdx.x / kWarpThreads;
  smem[warp_id] = sum1;
  __syncthreads();
  if (warp_id == 0) {
    const auto tx = threadIdx.x;
    sum1 = warp::reduce_sum(tx < kNumWarps ? smem[tx] : 0.0f);
    smem[32] = math::rsqrt(sum1 / kDim + eps);
  }
  __syncthreads();
  const float norm1 = smem[32];

  // --- Fused: mid = rmsnorm1(input)*weight1 + residual ---
  // Also compute sum-of-squares of mid for the second norm.
  // After this loop, input_vec / w1_vec / res_vec are dead.
  Storage mid_vec;
  float sum2 = 0.0f;
#pragma unroll
  for (auto j = 0u; j < 4u; ++j) {
    const auto [ix, iy] = cast<fp32x2_t>(input_vec[j]);
    const auto [w1x, w1y] = cast<fp32x2_t>(w1_vec[j]);
    const auto [rx, ry] = cast<fp32x2_t>(res_vec[j]);
    const float mx = ix * norm1 * w1x + rx;
    const float my = iy * norm1 * w1y + ry;
    sum2 += mx * mx + my * my;
    mid_vec[j] = cast<Float2>(fp32x2_t{mx, my});
  }

  // Store mid (the new residual)
  const auto res_out_ptr = pointer::offset<Float>(residual_out, row);
  gmem.store(res_out_ptr, mid_vec);

  // --- Second RMSNorm: CTA reduce on mid ---
  sum2 = warp::reduce_sum(sum2);
  smem[warp_id] = sum2;
  __syncthreads();
  if (warp_id == 0) {
    const auto tx = threadIdx.x;
    sum2 = warp::reduce_sum(tx < kNumWarps ? smem[tx] : 0.0f);
    smem[32] = math::rsqrt(sum2 / kDim + eps);
  }
  __syncthreads();
  const float norm2 = smem[32];

  // --- Output = rmsnorm2(mid) * weight2 ---
  Storage out_vec;
#pragma unroll
  for (auto j = 0u; j < 4u; ++j) {
    const auto [mx, my] = cast<fp32x2_t>(mid_vec[j]);
    const auto [w2x, w2y] = cast<fp32x2_t>(w2_vec[j]);
    out_vec[j] = cast<Float2>(fp32x2_t{mx * norm2 * w2x, my * norm2 * w2y});
  }

  const auto out_ptr = pointer::offset<Float>(output, row);
  gmem.store(out_ptr, out_vec);

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kDim, bool kUsePDL, typename DType>
struct FusedDoubleRMSNormKernel {
  static_assert(kDim > 256 && kDim % 256 == 0 && kDim <= 8192, "Hidden size must be a multiple of 256 in (256, 8192]");
  static_assert(std::is_same_v<DType, fp16_t> || std::is_same_v<DType, bf16_t>, "Only fp16 and bf16 are supported");
  static constexpr auto kernel = fused_double_rmsnorm_cta<kDim, kUsePDL, DType>;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView weight1,
      const tvm::ffi::TensorView weight2,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView residual_out,
      float eps) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto device = SymbolicDevice{};
    D.set_value(kDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({D})  //
        .with_dtype<DType>()
        .with_device(device)
        .verify(weight1)
        .verify(weight2);
    TensorMatcher({N, D})  //
        .with_dtype<DType>()
        .with_device(device)
        .verify(input)
        .verify(residual)
        .verify(output)
        .verify(residual_out);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = FusedDoubleRMSNormParams{
        .input = input.data_ptr(),
        .weight1 = weight1.data_ptr(),
        .weight2 = weight2.data_ptr(),
        .residual_in = residual.data_ptr(),
        .output = output.data_ptr(),
        .residual_out = residual_out.data_ptr(),
        .num_tokens = num_tokens,
        .eps = eps,
    };

    static constexpr uint32_t kNumThreads = (kDim / 256) * device::kWarpThreads;
    LaunchKernel(num_tokens, kNumThreads, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
