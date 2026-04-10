/// \file deepseek/dsv3_indexer.cuh
/// \brief TVM-FFI bindings for the DSV3 indexer + topk kernels.

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include "dsv3_indexer_sm100.cuh"
#include "dsv3_topk_sm100.cuh"
#include <cstdint>
#include <cuda_runtime.h>

using tvm::ffi::TensorView;

// ---------------------------------------------------------------------------
// Indexer binding
// ---------------------------------------------------------------------------

void dsv3_indexer(
    TensorView q,
    TensorView k_cache,
    TensorView q_scale,
    TensorView seq_lens,
    TensorView block_table,
    TensorView metadata,
    TensorView logits) {
  using namespace host;
  const int64_t batch_size = q.size(0);

  RuntimeCheck(q.ndim() == 3 && q.size(1) == 64 && q.size(2) == 128, "q must be [batch, 64, 128]");
  RuntimeCheck(
      k_cache.ndim() == 4 && k_cache.size(1) == 64 && k_cache.size(2) == 1 && k_cache.size(3) == 132,
      "k_cache must be [num_pages, 64, 1, 132]");
  RuntimeCheck(
      q_scale.ndim() == 2 && q_scale.size(0) == batch_size && q_scale.size(1) == 64, "q_scale must be [batch, 64]");
  RuntimeCheck(seq_lens.ndim() == 1 && seq_lens.size(0) == batch_size, "seq_lens must be [batch]");
  RuntimeCheck(
      block_table.ndim() == 2 && block_table.size(0) == batch_size, "block_table must be [batch, max_num_pages]");
  RuntimeCheck(metadata.ndim() == 2 && metadata.size(1) == 4, "metadata must be [num_sms, 4]");
  RuntimeCheck(logits.ndim() == 2 && logits.size(0) == batch_size, "logits must be [batch, logit_stride]");

  const int num_pages = static_cast<int>(k_cache.size(0));
  const int num_sms = static_cast<int>(metadata.size(0));
  const int page_table_stride = static_cast<int>(block_table.stride(0));
  const int logit_stride = static_cast<int>(logits.stride(0));

  cudaStream_t stream = LaunchKernel::resolve_device(q.device());

  sglang::dsv3::launch_dsv3_indexer(
      reinterpret_cast<uint8_t*>(q.data_ptr()),
      reinterpret_cast<uint8_t*>(k_cache.data_ptr()),
      static_cast<float*>(q_scale.data_ptr()),
      static_cast<int*>(seq_lens.data_ptr()),
      static_cast<int*>(block_table.data_ptr()),
      static_cast<float*>(logits.data_ptr()),
      reinterpret_cast<sglang::dsv3::ScheduleMetadata*>(metadata.data_ptr()),
      num_pages,
      static_cast<int>(batch_size),
      num_sms,
      page_table_stride,
      logit_stride,
      stream);

  RuntimeDeviceCheck();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dsv3_indexer, dsv3_indexer);

// ---------------------------------------------------------------------------
// TopK binding
// ---------------------------------------------------------------------------

void dsv3_topk(
    TensorView logits,
    TensorView seq_lens,
    TensorView out_indices,
    TensorView overflow_buf,
    int64_t K,
    int64_t num_clusters) {
  using namespace host;
  const int batch_size = static_cast<int>(logits.size(0));

  RuntimeCheck(logits.ndim() == 2, "logits must be [batch, stride]");
  RuntimeCheck(seq_lens.ndim() == 1 && seq_lens.size(0) == batch_size, "seq_lens must be [batch]");
  RuntimeCheck(
      out_indices.ndim() == 2 && out_indices.size(0) == batch_size && out_indices.size(1) >= K,
      "out_indices must be [batch, >=K]");
  RuntimeCheck(overflow_buf.ndim() == 2 && overflow_buf.size(0) == batch_size, "overflow_buf must be [batch, ...]");
  RuntimeCheck(
      num_clusters == 1 || num_clusters == 2 || num_clusters == 4 || num_clusters == 8,
      "num_clusters must be 1, 2, 4, or 8");
  RuntimeCheck(K >= 1 && K <= 2048, "K must be in [1, 2048]");

  const int logits_stride = static_cast<int>(logits.stride(0));
  const int out_stride = static_cast<int>(out_indices.stride(0));
  const int ov_total = static_cast<int>(overflow_buf.size(1));
  const int ov_stride = ov_total / (static_cast<int>(num_clusters) * 4);

  cudaStream_t stream = LaunchKernel::resolve_device(logits.device());

  sglang::dsv3::launch_dsv3_topk(
      static_cast<const float*>(logits.data_ptr()),
      static_cast<const int*>(seq_lens.data_ptr()),
      static_cast<int*>(out_indices.data_ptr()),
      static_cast<int*>(overflow_buf.data_ptr()),
      static_cast<int>(K),
      logits_stride,
      out_stride,
      ov_stride,
      batch_size,
      static_cast<int>(num_clusters),
      stream);

  RuntimeDeviceCheck();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dsv3_topk, dsv3_topk);
