#undef Py_LIMITED_API
#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/ops/from_blob.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_map>
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

struct MetaData {
  cudaIpcMemHandle_t handle;
  std::array<int, 4> shape;
  std::array<int, 4> stride;
  c10::Device device;
  c10::Layout layout;
  c10::ScalarType dtype;
  int dim;
  int64_t offset;
};

auto share_ipc_tensor(torch::Tensor tensor) -> MetaData {
  auto [offset, str] = at::cuda::CUDACachingAllocator::shareIpcHandle(tensor.storage().mutable_data());
  std::cerr << (int)str[0] << " " << (int)str[1] << " " << str.size() << std::endl;
  if (str[1] != 'c') {
    throw std::runtime_error("Invalid IPC handle string format: expected 'c' at position 1.");
  }
  if (str.size() != 2 + sizeof(cudaIpcMemHandle_t)) {
    std::cerr << "Invalid IPC handle string size: " << str.size() << std::endl;
    throw std::runtime_error("Invalid IPC handle string format.");
  }

  auto result = MetaData{
      .device = tensor.device(),
      .layout = tensor.layout(),
      .dtype = tensor.scalar_type(),
  };

  std::memcpy(&result.handle, str.data() + 2, sizeof(cudaIpcMemHandle_t));
  auto shape = tensor.sizes().vec();
  auto stride = tensor.strides().vec();
  if (shape.size() >= 4 || stride.size() >= 4) {
    throw std::runtime_error("Tensor shape or stride exceeds 4 dimensions.");
  }

  std::copy(shape.begin(), shape.end(), result.shape.begin());
  std::copy(stride.begin(), stride.end(), result.stride.begin());
  result.dim = static_cast<int>(tensor.dim());
  result.offset = offset;
  return result;
}

struct CompareEQ {
  bool operator()(const cudaIpcMemHandle_t& lhs, const cudaIpcMemHandle_t& rhs) const {
    return std::memcmp(&lhs, &rhs, sizeof(cudaIpcMemHandle_t)) == 0;
  }
};

struct HashIpcHandle {
  std::size_t operator()(const cudaIpcMemHandle_t& handle) const {
    std::size_t hash = 0;
    for (auto c : handle.reserved) {
      hash = (hash * 31) ^ static_cast<std::size_t>(c);
    }
    return hash;
  }
};

std::unordered_map<cudaIpcMemHandle_t, void*, HashIpcHandle, CompareEQ> ipc_tensor_cache;

auto open_ipc_tensor(const MetaData& meta) -> torch::Tensor {
  void* ptr = nullptr;
  if (auto& ref = ipc_tensor_cache[meta.handle]) {
    ptr = ref;
  } else {
    std::cerr << "Opening IPC handle: \n";
    C10_CUDA_CHECK(cudaIpcOpenMemHandle(&ptr, meta.handle, cudaIpcMemLazyEnablePeerAccess));
    ref = ptr;  // Cache the pointer for future use
  }
  auto options = torch::TensorOptions().device(meta.device).layout(meta.layout).dtype(meta.dtype).requires_grad(false);
  auto shape = std::vector<int64_t>(meta.shape.begin(), meta.shape.begin() + meta.dim);
  auto stride = std::vector<int64_t>(meta.stride.begin(), meta.stride.begin() + meta.dim);
  ptr = static_cast<char*>(ptr) + meta.offset;
  auto tensor = at::from_blob(ptr, shape, stride, options);
  tensor.set_requires_grad(false);
  return tensor;
}

auto share_ipc_bytes(torch::Tensor tensor) -> pybind11::bytes {
  auto meta = share_ipc_tensor(tensor);
  char buffer[sizeof(MetaData) + 1] = {0};
  std::memcpy(buffer, &meta, sizeof(MetaData));
  return pybind11::bytes(buffer, sizeof(MetaData));
}

auto open_ipc_bytes(std::string buffer) -> torch::Tensor {
  auto meta = MetaData{.device = c10::kCPU, .layout = c10::kStrided, .dtype = c10::kFloat};
  std::memcpy(&meta, buffer.data(), sizeof(MetaData));
  return open_ipc_tensor(meta);
}

PYBIND11_MODULE(cuda_utils, m) {
  m.attr("__name__") = "sgl_kernel.cuda_utils";
  m.def("get_sm_count", &get_sm_count);
  m.def("split_by_count", &split_by_count);
  m.def("set_cublas_sm", &set_cublas_sm);
  m.def("share_ipc_tensor", &share_ipc_bytes);
  m.def("open_ipc_tensor", &open_ipc_bytes);
}

}  // namespace
