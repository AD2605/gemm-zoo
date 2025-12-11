#ifndef NVIDIA_UTILS_COMPARE_CUH
#define NVIDIA_UTILS_COMPARE_CUH

#include "defines.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace utils {

template <typename T>
__global__ void compare_results_kernel(const T* output, const T* reference,
                                       std::size_t num_elements,
                                       int* is_different) {
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  float compare_threshold = 1e-2;
  if constexpr (!std::is_same_v<T, float>) {
    compare_threshold = 0.5;
  }
  for (; idx < num_elements; idx += gridDim.x * blockDim.x) {
    if (abs(static_cast<float>(output[idx]) -
            static_cast<float>(reference[idx])) > compare_threshold) {
      // #ifdef DEBUG
      printf("%f %f %d \n", static_cast<float>(output[idx]),
             static_cast<float>(reference[idx]), idx);
      // #endif
      *is_different = 1;
    }
  }
}

template <typename T>
void compare_results(const T* output, const T* reference,
                     std::size_t num_elements, cudaStream_t stream) {
  int* is_different;
  cudaMallocManaged(&is_different, sizeof(int));
  *is_different = 0;
  dim3 blockDim(32, 1, 1);
  dim3 gridDim((num_elements + 32 - 1) / 32, 1, 1);
  compare_results_kernel<<<gridDim, blockDim, 0, stream>>>(
      output, reference, num_elements, is_different);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaStreamSynchronize(stream));
  if (*is_different) {
    throw std::runtime_error("verification failed");
  }
}
}  // namespace utils

#endif
