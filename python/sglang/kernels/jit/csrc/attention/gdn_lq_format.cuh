// LeapQuant 8-bit GDN state format (Qwen3.5 GDN: K = V = 128, r = 4 compensators).
//
// One format block per (layer, slot, value head), aliasing that head's 64 KB fp32 temporal-pool region
// (option A). Byte offsets inside the head region:
//   Z int8 [V][K] | s fp32 [V] | c fp32 [K] | K~ fp16 [4][K] | U~ fp16 [4][V] | u fp16 [C][V]
//   | k^ fp16 [C][K] (per KEY head: valid in the region of the even value head 2i of key head i)
//   | G fp32 [C] | l int32, lim int32 (replicated in EVERY value head of the (layer, slot), equal
//     copies; lim 0 = P)
// Base B = diag(s) Z diag(c) + U~^T K~; record j (0-based) = window position j + 1;
// S_l = e^{G_l} B + sum_{j < l} u_j (k^_j e^{G_l - G_j})^T, G_l = G[l - 1] (0 when l = 0).
#pragma once

#include <cstdint>

namespace sglang::gdn_lq {

inline constexpr uint32_t kDim = 128;                    // K = V
inline constexpr uint32_t kRank = 4;                     // compensator tokens
inline constexpr uint32_t kHeadBytes = kDim * kDim * 4;  // one fp32 head region of the temporal pool

template <uint32_t kC>
struct Layout {
  static constexpr uint32_t kZ = 0;
  static constexpr uint32_t kS = kZ + kDim * kDim;
  static constexpr uint32_t kCc = kS + kDim * 4;
  static constexpr uint32_t kKt = kCc + kDim * 4;
  static constexpr uint32_t kUt = kKt + kRank * kDim * 2;
  static constexpr uint32_t kU = kUt + kRank * kDim * 2;
  static constexpr uint32_t kKh = kU + kC * kDim * 2;
  static constexpr uint32_t kG = kKh + kC * kDim * 2;
  static constexpr uint32_t kL = kG + (kC * 4 + 15) / 16 * 16;
  static constexpr uint32_t kLim = kL + 4;
  static constexpr uint32_t kBytes = kL + 16;
  static_assert(kBytes <= kHeadBytes);
};

}  // namespace sglang::gdn_lq
