/// \file deepseek/sm100a_utils.cuh
/// \brief Reusable SM100a (Blackwell) primitives: TMA, tcgen05 MMA, tensor memory,
///        mbarrier synchronization, and tensor-map encoding helpers.
///
/// These utilities are architecture-specific but kernel-agnostic ??? they can be
/// shared across any SM100a kernel that uses tcgen05 MMA or TMA bulk copies.

#pragma once

#include <sgl_kernel/utils.cuh>

#include <cuda/ptx>

#include <cstdint>
#include <cudaTypedefs.h>
#include <cuda_fp8.h>
#include <stdexcept>
#include <string>

namespace sglang::dsv3 {

// ---------------------------------------------------------------------------
// mbarrier primitives (wrapping cuda::ptx where possible)
// ---------------------------------------------------------------------------

/// Blocking parity wait: spins on cuda::ptx::mbarrier_try_wait_parity
/// with acquire.cta semantics until the barrier flips to the given phase.
SGL_DEVICE void mbarrier_wait(uint64_t* addr, uint32_t phase) {
  while (!cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_acquire, cuda::ptx::scope_cta, addr, phase)) {
  }
}

SGL_DEVICE void mbarrier_init(uint64_t* addr, uint32_t count) {
  cuda::ptx::mbarrier_init(addr, count);
}

/// Arrive with expected TX byte count (for TMA completion tracking).
SGL_DEVICE void mbarrier_arrive_expect_tx(uint64_t* addr, uint32_t tx_count) {
  cuda::ptx::mbarrier_arrive_expect_tx(
      cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, addr, tx_count);
}

SGL_DEVICE void mbarrier_arrive(uint64_t* addr) {
  cuda::ptx::mbarrier_arrive(cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, addr);
}

// ---------------------------------------------------------------------------
// SM100a: elect_sync (leader election)
// ---------------------------------------------------------------------------

SGL_DEVICE uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred %%px;\n\t"
      "elect.sync _|%%px, %1;\n\t"
      "@%%px mov.s32 %0, 1;\n\t"
      "}"
      : "+r"(pred)
      : "r"(0xFFFFFFFF));
  return pred;
}

// ---------------------------------------------------------------------------
// TMA descriptor prefetch
// ---------------------------------------------------------------------------

/// Prefetch a TMA tensor map descriptor into cache (SM90+).
/// From CUTLASS: cute/arch/copy_sm90_desc.hpp
SGL_DEVICE void prefetch_tma_descriptor(const void* desc_ptr) {
  uint64_t addr = reinterpret_cast<uint64_t>(desc_ptr);
  asm volatile("prefetch.tensormap [%0];" : : "l"(addr) : "memory");
}

// ---------------------------------------------------------------------------
// TMA gmem -> smem helpers
// ---------------------------------------------------------------------------

SGL_DEVICE void tma_1d_gmem2smem(const void* src, void* dst, int num_bytes, uint64_t* mbar) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];" ::"l"(dst),
      "l"(src),
      "r"(num_bytes),
      "l"(mbar)
      : "memory");
}

SGL_DEVICE void tma_1d_gmem2smem(void* dst, const void* src, int num_bytes, uint64_t* mbar) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];" ::"l"(dst),
      "l"(src),
      "r"(num_bytes),
      "l"(mbar)
      : "memory");
}

SGL_DEVICE void tma_3d_gmem2smem(void* dst, const void* tmap_ptr, int x, int y, int z, uint64_t* mbar_addr) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_"
      "tx::bytes.cta_group::1 "
      "[%0], [%1, {%2, %3, %4}], [%5];" ::"l"(dst),
      "l"(tmap_ptr),
      "r"(x),
      "r"(y),
      "r"(z),
      "l"(mbar_addr)
      : "memory");
}

// ---------------------------------------------------------------------------
// tcgen05 MMA (SM100a)
// ---------------------------------------------------------------------------

SGL_DEVICE void tcgen05_mma_f8(int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc, int enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
      "}" ::"r"(taddr),
      "l"(a_desc),
      "l"(b_desc),
      "r"(i_desc),
      "r"(enable_input_d));
}

SGL_DEVICE constexpr uint64_t desc_encode(uint64_t x) {
  return (x & 0x3'FFFFULL) >> 4ULL;
}

// ---------------------------------------------------------------------------
// tcgen05 load variants (SM100a tensor memory)
// ---------------------------------------------------------------------------

SGL_DEVICE void tcgen05_ld_32x32b_x2(int addr, float (&tmp)[2]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0, %1}, [%2];" : "=f"(tmp[0]), "=f"(tmp[1]) : "r"(addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

SGL_DEVICE void tcgen05_ld_32x32b_x4(int addr, float (&tmp)[4]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
               : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3])
               : "r"(addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

SGL_DEVICE void tcgen05_ld_32x32b_x8(int addr, float (&tmp)[8]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]), "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
      : "r"(addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

SGL_DEVICE void tcgen05_ld_32x32b_x16(int addr, float (&tmp)[16]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, "
      "%6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
      : "=f"(tmp[0]),
        "=f"(tmp[1]),
        "=f"(tmp[2]),
        "=f"(tmp[3]),
        "=f"(tmp[4]),
        "=f"(tmp[5]),
        "=f"(tmp[6]),
        "=f"(tmp[7]),
        "=f"(tmp[8]),
        "=f"(tmp[9]),
        "=f"(tmp[10]),
        "=f"(tmp[11]),
        "=f"(tmp[12]),
        "=f"(tmp[13]),
        "=f"(tmp[14]),
        "=f"(tmp[15])
      : "r"(addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

template <int WIDTH>
SGL_DEVICE void tcgen05_ld_32x32b(int addr, float (&tmp)[WIDTH]) {
  if constexpr (WIDTH == 2) {
    tcgen05_ld_32x32b_x2(addr, tmp);
  } else if constexpr (WIDTH == 4) {
    tcgen05_ld_32x32b_x4(addr, tmp);
  } else if constexpr (WIDTH == 8) {
    tcgen05_ld_32x32b_x8(addr, tmp);
  } else if constexpr (WIDTH == 16) {
    tcgen05_ld_32x32b_x16(addr, tmp);
  } else {
    static_assert(WIDTH == 2, "WIDTH must be 2, 4, 8, or 16");
  }
}

SGL_DEVICE void tcgen05_alloc(int ncols, int* tmem_addr) {
  const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(addr), "r"(ncols));
}

SGL_DEVICE void tcgen05_dealloc(int ncols, int taddr) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols));
}

SGL_DEVICE void tcgen05_mma_arrive(void* addr) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"l"(addr) : "memory");
}

// ---------------------------------------------------------------------------
// Shared-memory address conversion
// ---------------------------------------------------------------------------

template <typename T>
SGL_DEVICE int cvt_addr(T* addr) {
  return static_cast<int>(__cvta_generic_to_shared(addr));
}

// ---------------------------------------------------------------------------
// Kernel attribute setup (one-shot cudaFuncSetAttribute)
// ---------------------------------------------------------------------------

template <auto* kernel_func, size_t smem_bytes>
void setup_kernel_smem_once() {
  static const cudaError_t result = []() -> cudaError_t {
    return cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  }();
  if (result != cudaSuccess) {
    throw std::runtime_error(
        std::string("cudaFuncSetAttribute(MaxDynamicSharedMemorySize) failed: ") + cudaGetErrorString(result));
  }
}

// ---------------------------------------------------------------------------
// TMA tensor-map helpers (host-side)
// ---------------------------------------------------------------------------

template <int rank>
inline void
init_tensormap_nd(CUtensorMap* tmap, uint8_t* ptr, uint64_t* globalDim, uint64_t* globalStrides, uint32_t* boxDim) {
  uint32_t elem_strides[rank];
  for (int i = 0; i < rank; ++i)
    elem_strides[i] = 1;
  auto err = cuTensorMapEncodeTiled(
      tmap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
      rank,
      (void*)ptr,
      globalDim,
      globalStrides,
      boxDim,
      elem_strides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (err != CUDA_SUCCESS) {
    const char* err_str = nullptr;
    cuGetErrorString(err, &err_str);
    throw std::runtime_error(std::string("cuTensorMapEncodeTiled failed: ") + (err_str ? err_str : "unknown error"));
  }
}

}  // namespace sglang::dsv3

// TYPE CHECKING only
extern "C" SGL_DEVICE float2 __ffma2_rn(float2 a, float2 b, float2 c);
