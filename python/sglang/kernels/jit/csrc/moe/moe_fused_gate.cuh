#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cfloat>
#include <type_traits>

namespace sglang {

struct MoeFusedGateParams {
  const void* scores;
  const void* bias;
  float* out_weights;
  int32_t* out_indices;
  float routed_scaling_factor;
  uint32_t num_tokens;
  uint32_t num_fused_shared_experts;
  bool renormalize;
  bool apply_scale;
};

enum class ScoringFunc {
  SIGMOID,
  SQRTSOFTPLUS,
};

template <typename ScoreT, typename BiasT, uint32_t E, uint32_t K, ScoringFunc kScoringFunc_>
struct MoeWarpTopKImpl {
  static_assert(E > 0);
  static_assert(K > 0 && K <= E, "cannot select more experts than exist");
  // The output stage is warp-based: lane i emits slot i, so a token's whole
  // top-k has to fit in one warp. Fused shared experts take the slots past K,
  // so the host checks K + num_fused_shared_experts against the same bound.
  static_assert(K <= device::kWarpThreads, "top-k must fit in one warp");

  using scores_t = ScoreT;
  static constexpr bool kHasBias = !std::is_void_v<BiasT>;
  // A void bias still needs a concrete element type for the register array the
  // no-bias path leaves dead; scores_t keeps every DTypeTrait lookup valid.
  using bias_t = std::conditional_t<kHasBias, BiasT, ScoreT>;

  static constexpr ScoringFunc kScoringFunc = kScoringFunc_;
  static constexpr uint32_t kNumExperts = E;
  static constexpr uint32_t kTopK = K;
  static constexpr uint32_t kPerLane = host::div_ceil(E, device::kWarpThreads);
  static constexpr uint32_t kVecSize = [] {
    constexpr uint32_t kMaxVecSize = device::kMaxVecBytes / std::max(sizeof(scores_t), sizeof(bias_t));
    for (uint32_t v = kMaxVecSize; v > 1; v /= 2) {
      // Dividing E keeps a vector from straddling the row end, so one bound
      // check covers all of its elements; dividing kPerLane keeps the wider
      // load from padding beyond what E already forces.
      if (E % v == 0 && kPerLane % v == 0) return v;
    }
    return 1u;
  }();
  static constexpr uint32_t kLocalVecs = kPerLane / kVecSize;
  static constexpr bool kIsPadded = device::kWarpThreads * kPerLane != E;
  // -FLT_MAX, not -1: an activation is >= 0 but the bias added to it is
  // unbounded below, so -1 would let a real expert lose to a padding slot.
  static constexpr float kPadScore = -FLT_MAX;

  /// \brief Whether lane `lane_id`'s `vec_id`-th vector holds real experts.
  static constexpr bool in_bound(uint32_t lane_id, uint32_t vec_id) {
    if constexpr (!kIsPadded) {
      return true;
    } else {
      return lane_id * kPerLane + vec_id * kVecSize < E;
    }
  }
};

template <ScoringFunc kScoringFunc>
SGL_DEVICE float act_one(float x) {
  if constexpr (kScoringFunc == ScoringFunc::SIGMOID) {
    return 1.0f / (1.0f + expf(-x));
  } else {
    static_assert(kScoringFunc == ScoringFunc::SQRTSOFTPLUS);
    // MUFU.LG2's error is absolute, so recovering log1p from log degrades as u
    // approaches 1 -- 1.7e-2 off fp64 by |x| = 16. Past |x| = 4 the three-term
    // series is within z^3/4 instead, and it subsumes the u == 1 guard:
    // |x| <= 4 keeps z >= 0.018.
    const float ax = fabsf(x);
    const float z = expf(-ax);
    const float u = 1.0f + z;
    const float series = z * fmaf(z, fmaf(z, 1.0f / 3.0f, -0.5f), 1.0f);
    const float log1p_z = ax > 4.0f ? series : z * logf(u) / (u - 1.0f);
    const float softplus = fmaxf(x, 0.0f) + log1p_z;
    return sqrtf(softplus);
  }
}

/**
 * \brief Top-K over one warp's registers, K dependent rounds.
 *
 * Each round takes the warp max of `biased`, resolves the owning lane from a
 * ballot, and broadcasts that lane's weight and expert id. Broadcasting beats
 * the masked sum-reduction the Triton router uses: one SHFL instead of a
 * five-step butterfly per round.
 */
template <typename Trait>
SGL_DEVICE void warp_topk(
    float (&biased)[Trait::kPerLane],
    const float (&activated)[Trait::kPerLane],
    uint32_t lane_id,
    float& out_weight,
    int32_t& out_index,
    float& routed_sum) {
  constexpr uint32_t L = Trait::kPerLane;
  constexpr uint32_t K = Trait::kTopK;
  constexpr uint32_t kMask = 0xffffffffu;

#pragma unroll
  for (uint32_t k = 0; k < K; ++k) {
    float local_max = biased[0];
#pragma unroll
    for (uint32_t j = 1; j < L; ++j) {
      local_max = fmaxf(local_max, biased[j]);
    }

    const auto max_score = device::warp::reduce_max(local_max);

    uint32_t slot = L;
    float cand_w = activated[0];
#pragma unroll
    for (uint32_t j = 0; j < L; ++j) {
      const auto hit = (biased[j] == local_max);
      slot = hit ? j : slot;
      cand_w = hit ? activated[j] : cand_w;
    }
    const auto eq = __ballot_sync(kMask, local_max == max_score);
    const auto win_lane = static_cast<uint32_t>(__ffs(eq) - 1);
    const auto mask_slot = lane_id == win_lane ? slot : L;

#pragma unroll
    for (uint32_t j = 0; j < L; ++j) {
      biased[j] = (j == mask_slot) ? -FLT_MAX : biased[j];
    }

    const auto w = __shfl_sync(kMask, cand_w, win_lane);
    const auto e = __shfl_sync(kMask, static_cast<int32_t>(lane_id * L + slot), win_lane);
    routed_sum += w;
    out_weight = (lane_id == k) ? w : out_weight;
    out_index = (lane_id == k) ? e : out_index;
  }
}

template <typename Trait, bool kUsePDL>
__global__ void moe_fused_gate_kernel(const MoeFusedGateParams params) {
  using namespace device;
  using T = typename Trait::scores_t;
  using B = typename Trait::bias_t;
  constexpr uint32_t E = Trait::kNumExperts;
  constexpr uint32_t K = Trait::kTopK;
  constexpr uint32_t V = Trait::kVecSize;
  constexpr uint32_t N = Trait::kLocalVecs;
  constexpr uint32_t L = Trait::kPerLane;

  const auto lane_id = threadIdx.x;
  const auto work_id = blockIdx.x * blockDim.y + threadIdx.y;
  if (work_id >= params.num_tokens) return;

  PDLWaitPrimary<kUsePDL>();

  const auto scores_ptr = static_cast<const T*>(params.scores) + static_cast<size_t>(work_id) * E;
  AlignedVector<T, V> scores_vecs[N];
#pragma unroll
  for (uint32_t i = 0; i < N; ++i) {
    // A padded vector is never read, so skip its load outright. in_bound folds
    // to a literal true when the warp covers E exactly, leaving no predicate.
    if (Trait::in_bound(lane_id, i)) {
      scores_vecs[i].load(scores_ptr, lane_id * N + i);
    }
  }

  AlignedVector<B, V> bias_vecs[N];
  if constexpr (Trait::kHasBias) {
    const auto bias_ptr = static_cast<const B*>(params.bias);
#pragma unroll
    for (uint32_t i = 0; i < N; ++i) {
      if (Trait::in_bound(lane_id, i)) {
        bias_vecs[i].load(bias_ptr, lane_id * N + i);
      }
    }
  }

  float activated[L];
  float biased[L];
#pragma unroll
  for (uint32_t i = 0; i < N; ++i) {
    if (Trait::in_bound(lane_id, i)) {
#pragma unroll
      for (uint32_t j = 0; j < V; ++j) {
        // fmaxf(NaN, 0) returns 0, which is how a NaN logit leaves the ranking.
        const float a = fmaxf(act_one<Trait::kScoringFunc>(cast<float>(scores_vecs[i][j])), 0.0f);
        activated[i * V + j] = a;
        if constexpr (Trait::kHasBias) {
          biased[i * V + j] = a + cast<float>(bias_vecs[i][j]);
        } else {
          biased[i * V + j] = a;
        }
      }
    } else {
#pragma unroll
      for (uint32_t j = 0; j < V; ++j) {
        activated[i * V + j] = 0.0f;
        biased[i * V + j] = Trait::kPadScore;
      }
    }
  }

  float out_weight = 0.0f;
  int32_t out_index = 0;
  float routed_sum = 0.0f;
  warp_topk<Trait>(biased, activated, lane_id, out_weight, out_index, routed_sum);
  PDLTriggerSecondary<kUsePDL>();

  // Only the top-k rounds need K at compile time; the fused shared experts ride
  // along at runtime in the slots past K, never entering the rounds themselves.
  const auto topk_total = K + params.num_fused_shared_experts;
  if (lane_id >= K) {
    out_weight = routed_sum / params.routed_scaling_factor;
    out_index = static_cast<int32_t>(E + lane_id - K);
  }
  if (params.renormalize) {
    out_weight /= (routed_sum > 0.0f ? routed_sum : 1.0f);
  }
  if (params.apply_scale) {
    out_weight *= params.routed_scaling_factor;
  }
  if (lane_id < topk_total) {
    const auto out_offset = static_cast<size_t>(work_id) * topk_total + lane_id;
    params.out_weights[out_offset] = out_weight;
    params.out_indices[out_offset] = out_index;
  }
}

/**
 * \brief Warp-per-token fused router: activation (+ bias) + top-k (+ renorm).
 *
 * The shape and the dtypes are template parameters so Python instantiates
 * exactly the configurations a model uses; there is no runtime dispatch and
 * nothing unused gets compiled.
 *
 * \tparam ScoreT       Score element type, e.g. float.
 * \tparam BiasT        Bias element type, or void for an unbiased router.
 * \tparam E            Expert count; any value, padded up to a warp multiple.
 * \tparam K            Routed experts selected per token; K plus the fused
 *                     shared experts must fit in one warp.
 * \tparam kScoring     SIGMOID or SQRTSOFTPLUS.
 * \tparam kUsePDL      Emit the PDL wait/trigger pair (SM90+).
 * \param scores        [num_tokens, E] logits, contiguous.
 * \param bias          [E] ranking bias, or none; the emitted weight stays bias-free.
 * \param weights       [num_tokens, K + num_fused_shared_experts] fp32 weights.
 * \param indices       [num_tokens, K + num_fused_shared_experts] int32 expert ids.
 * \param num_fused_shared_experts  Shared experts appended after the routed ones,
 *                     taking ids E.. and weight routed_sum / routed_scaling_factor.
 */
template <typename ScoreT, typename BiasT, uint32_t E, uint32_t K, ScoringFunc kScoring, bool kUsePDL>
void moe_fused_gate(
    tvm::ffi::TensorView scores,
    tvm::ffi::Optional<tvm::ffi::TensorView> bias,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView indices,
    bool renormalize,
    float routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output,
    int64_t num_fused_shared_experts) {
  using namespace host;
  using Trait = MoeWarpTopKImpl<ScoreT, BiasT, E, K, kScoring>;

  auto M = SymbolicSize{"num_tokens"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();
  TensorMatcher({M, E}).with_dtype<ScoreT>().with_device(device_).verify(scores);
  CHECK_HOST(num_fused_shared_experts >= 0) << "moe_fused_gate_v2: num_fused_shared_experts is negative";
  const auto num_shared = static_cast<uint32_t>(num_fused_shared_experts);
  const auto topk_total = K + num_shared;
  CHECK_HOST(topk_total <= device::kWarpThreads)
      << "moe_fused_gate_v2: topk " << K << " + " << num_shared << " fused shared experts exceeds the "
      << device::kWarpThreads << " output slots one warp can emit";
  TensorMatcher({M, topk_total}).with_dtype<float>().with_device(device_).verify(weights);
  TensorMatcher({M, topk_total}).with_dtype<int32_t>().with_device(device_).verify(indices);

  CHECK_HOST(bias.has_value() == Trait::kHasBias)
      << "moe_fused_gate_v2: a bias tensor was " << (bias.has_value() ? "given" : "omitted") << " but BiasT is "
      << (Trait::kHasBias ? "not void" : "void");
  const void* bias_ptr = nullptr;
  if constexpr (Trait::kHasBias) {
    TensorMatcher({E}).with_dtype<BiasT>().with_device(device_).verify(bias.value());
    bias_ptr = bias.value().data_ptr();
  }

  const auto params = MoeFusedGateParams{
      .scores = scores.data_ptr(),
      .bias = bias_ptr,
      .out_weights = static_cast<float*>(weights.data_ptr()),
      .out_indices = static_cast<int32_t*>(indices.data_ptr()),
      .routed_scaling_factor = static_cast<float>(routed_scaling_factor),
      .num_tokens = static_cast<uint32_t>(M.unwrap()),
      .num_fused_shared_experts = num_shared,
      .renormalize = renormalize,
      .apply_scale = apply_routed_scaling_factor_on_output,
  };
  const auto num_warps = params.num_tokens <= 32 ? 1u : 4u;
  const dim3 block{device::kWarpThreads, num_warps, 1};
  LaunchKernel(div_ceil(params.num_tokens, num_warps), block, device_.unwrap())
      .enable_pdl(kUsePDL)(moe_fused_gate_kernel<Trait, kUsePDL>, params);
}

using enum ScoringFunc;

}  // namespace sglang
