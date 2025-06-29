#include <ATen/cuda/CUDAContextLight.h>

#include <cstddef>
#undef Py_LIMITED_API
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cstdint>
#include <tuple>
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
  CUDA_RT(cudaInitDevice(device, 0, 0));
  CUDA_DRV(cuDeviceGetDevResource(device, &sm_resource, CU_DEV_RESOURCE_TYPE_SM));
  return sm_resource.sm.smCount;
}

auto resource_to_context(int device, CUdevResource resource) -> CUgreenCtx {
  auto desc = CUdevResourceDesc{};
  CUDA_DRV(cuDevResourceGenerateDesc(&desc, &resource, CU_DEV_RESOURCE_TYPE_SM));
  auto ctx = CUgreenCtx{};
  CUDA_DRV(cuGreenCtxCreate(&ctx, desc, device, CU_GREEN_CTX_DEFAULT_STREAM));
  return ctx;
}

auto context_to_stream(CUgreenCtx ctx) -> std::int64_t {
  auto stream = CUstream{};
  CUDA_DRV(cuGreenCtxStreamCreate(&stream, ctx, CU_STREAM_NON_BLOCKING, 0));
  return reinterpret_cast<std::int64_t>(stream);
}

auto get_device_resource(int device) -> CUdevResource {
  auto input = CUdevResource{};
  CUDA_DRV(cuDeviceGetDevResource(device, &input, CU_DEV_RESOURCE_TYPE_SM));
  return input;
}

// split the resource into the needed number of SMs
auto split_resource(CUdevResource input, unsigned needed) -> std::tuple<CUdevResource, CUdevResource> {
  if (input.sm.smCount < needed) {
    throw std::invalid_argument("Not enough SMs available in the device resource.");
  }
  if (input.sm.smCount == needed) {
    return std::make_tuple(input, CUdevResource{});  // No split needed, return the original resource.
  }

  auto resources = CUdevResource{};
  auto remaining = CUdevResource{};
  unsigned nb_groups = 1;
  CUDA_DRV(cuDevSmResourceSplitByCount(&resources, &nb_groups, &input, &remaining, 0, needed));

  if (nb_groups != 1) {
    throw std::runtime_error("Failed to split SM resources into the requested number of groups.");
  }
  if (resources.sm.smCount != needed) {
    throw std::runtime_error("The number of SMs in the split resource does not match the requested count.");
  }

  return std::make_tuple(resources, remaining);
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

auto split_by_count(int device, std::vector<int> sm_counts) -> std::vector<std::int64_t> {
  CUDA_RT(cudaInitDevice(device, 0, 0));
  const auto device_guard = DeviceGuard{device};
  auto last_resource = get_device_resource(device);
  auto resources = std::vector<CUdevResource>{};
  resources.reserve(sm_counts.size());
  for (const auto& sm_count : sm_counts) {
    if (sm_count <= 0) {
      throw std::invalid_argument("SM count must be greater than zero.");
    }
    const auto [resource, remaining] = split_resource(last_resource, sm_count);
    resources.push_back(resource);
    // need to refresh the last resource for the next iteration
    if (remaining.sm.smCount == 0) continue;
    auto ctx = resource_to_context(device, remaining);
    CUDA_DRV(cuGreenCtxGetDevResource(ctx, &last_resource, CU_DEV_RESOURCE_TYPE_SM));
  }
  // cast resources to context streams
  auto contexts = std::vector<std::int64_t>{};
  contexts.reserve(resources.size());
  for (const auto& resource : resources) {
    contexts.push_back(context_to_stream(resource_to_context(device, resource)));
  }
  return contexts;
}

auto set_cublas_sm(std::size_t sm_target) -> void {
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  cublasSetSmCountTarget(handle, sm_target);
}

PYBIND11_MODULE(cuda_utils, m) {
  m.attr("__name__") = "sgl_kernel.cuda_utils";
  m.def("get_sm_count", &get_sm_count);
  m.def("split_by_count", &split_by_count);
  m.def("set_cublas_sm", &set_cublas_sm);
}

}  // namespace
