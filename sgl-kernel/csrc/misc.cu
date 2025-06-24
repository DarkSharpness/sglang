#undef Py_LIMITED_API
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace {

#define CUDA_DRV(expr)                                                                          \
  do {                                                                                          \
    CUresult res = (expr);                                                                      \
    if (res != CUDA_SUCCESS) {                                                                  \
      const char* err_str;                                                                      \
      cuGetErrorString(res, &err_str);                                                          \
      throw std::runtime_error(                                                                 \
          std::string("CUDA Driver error: ") + err_str + std::string(" at ") + __FILE__ + ":" + \
          std::to_string(__LINE__));                                                            \
    }                                                                                           \
  } while (0)

#define CUDA_RT(expr)                                                                                            \
  do {                                                                                                           \
    cudaError_t res = (expr);                                                                                    \
    if (res != cudaSuccess) {                                                                                    \
      throw std::runtime_error(                                                                                  \
          std::string("CUDA Runtime error: ") + cudaGetErrorString(res) + std::string(" at ") + __FILE__ + ":" + \
          std::to_string(__LINE__));                                                                             \
    }                                                                                                            \
  } while (0)

auto get_sm_count(int device) -> unsigned {
  CUdevResource sm_resource;
  CUDA_DRV(cuDeviceGetDevResource(device, &sm_resource, CU_DEV_RESOURCE_TYPE_SM));
  return sm_resource.sm.smCount;
}

auto init_green_context(int device, unsigned sm_count) -> std::tuple<CUgreenCtx, CUgreenCtx> {
  auto input = CUdevResource{};
  CUDA_DRV(cuDeviceGetDevResource(device, &input, CU_DEV_RESOURCE_TYPE_SM));
  if (input.sm.smCount < sm_count) {
    throw std::runtime_error("Not enough SMs available for the requested count.");
  }
  auto required = CUdevResource{};
  auto remaining = CUdevResource{};

  auto nb_groups = 1u;
  CUDA_DRV(cuDevSmResourceSplitByCount(&required, &nb_groups, &input, &remaining, 0, sm_count));
  if (nb_groups == 0) {
    throw std::runtime_error("Failed to split SM resources.");
  }

  auto required_ctx = CUgreenCtx{};
  auto remaining_ctx = CUgreenCtx{};

  // init resource descriptor
  auto overall_desc = CUdevResourceDesc{};
  CUDA_DRV(cuDevResourceGenerateDesc(&overall_desc, &required, CU_DEV_RESOURCE_TYPE_SM));
  CUDA_DRV(cuGreenCtxCreate(&required_ctx, overall_desc, device, CU_GREEN_CTX_DEFAULT_STREAM));

  // init remaining resource descriptor
  overall_desc = CUdevResourceDesc{};
  CUDA_DRV(cuDevResourceGenerateDesc(&overall_desc, &remaining, CU_DEV_RESOURCE_TYPE_SM));
  CUDA_DRV(cuGreenCtxCreate(&remaining_ctx, overall_desc, device, CU_GREEN_CTX_DEFAULT_STREAM));

  return {required_ctx, remaining_ctx};
}

auto split_green_context(CUgreenCtx ctx, int device, unsigned groups, unsigned sm_each) -> std::vector<CUgreenCtx> {
  auto input = CUdevResource{};
  CUDA_DRV(cuGreenCtxGetDevResource(ctx, &input, CU_DEV_RESOURCE_TYPE_SM));

  if (input.sm.smCount != groups * sm_each) {
    throw std::invalid_argument("The number of SMs in the context does not match the requested groups.");
  }

  if (groups <= 1) {
    return {ctx};  // No splitting needed, return the original context.
  }

  auto nb_groups = groups;
  auto resources = std::vector<CUdevResource>(nb_groups);
  auto remaining = CUdevResource{};

  CUDA_DRV(cuDevSmResourceSplitByCount(resources.data(), &nb_groups, &input, &remaining, 0, sm_each));
  if (nb_groups != groups) {
    throw std::runtime_error("Failed to split SM resources into the requested number of groups.");
  }
  auto contexts = std::vector<CUgreenCtx>(groups);
  for (unsigned i = 0; i < groups; ++i) {
    auto desc = CUdevResourceDesc{};
    CUDA_DRV(cuDevResourceGenerateDesc(&desc, &resources[i], CU_DEV_RESOURCE_TYPE_SM));
    CUDA_DRV(cuGreenCtxCreate(&contexts[i], desc, device, CU_GREEN_CTX_DEFAULT_STREAM));
  }
  if (remaining.sm.smCount > 0) {
    throw std::runtime_error("There are remaining SMs that were not assigned to any context.");
  }
  return contexts;
}

auto stream_green_context(CUgreenCtx ctx) -> std::int64_t {
  CUstream stream;
  CUDA_DRV(cuGreenCtxStreamCreate(&stream, ctx, CU_STREAM_NON_BLOCKING, 0));
  return reinterpret_cast<std::int64_t>(stream);
}

struct DeviceGuard {
  DeviceGuard(int device) : m_old_device() {
    CUDA_RT(cudaGetDevice(&m_old_device));
    CUDA_RT(cudaSetDevice(device));
  }
  ~DeviceGuard() noexcept(false) {
    CUDA_RT(cudaSetDevice(m_old_device));
  }

 private:
  int m_old_device;
};

auto create_green_context(int device, int sm_needed, int num_split, int num_share) -> std::vector<std::int64_t> {
  if (num_split <= 0 || num_share <= 0) {
    throw std::invalid_argument("num_split and num_share must be greater than zero.");
  }

  CUDA_RT(cudaInitDevice(device, 0, 0));
  const auto device_guard = DeviceGuard{device};

  const auto [required, remaining] = init_green_context(device, sm_needed);
  const auto contexts = split_green_context(required, device, num_split, sm_needed / num_split);

  auto streams = std::vector<std::int64_t>{};
  streams.reserve(num_split * num_share + 1);  // +1 for the remaining context
  for (const auto& ctx : contexts) {
    for (int i = 0; i < num_share; ++i) {
      streams.push_back(stream_green_context(ctx));
    }
  }

  auto visited = std::unordered_set<std::int64_t>{};
  for (const auto& id : streams) {
    if (!visited.insert(id).second) {
      throw std::runtime_error("Duplicated stream ID detected.");
    }
  }

  // add the remaining context as a stream
  streams.push_back(stream_green_context(remaining));
  return streams;
}

PYBIND11_MODULE(cuda_utils, m) {
  // namespace py = pybind11;
  m.attr("__name__") = "sgl_kernel.cuda_utils";
  m.def("get_sm_count", &get_sm_count);
  m.def("create_green_context", &create_green_context);
}

}  // namespace
