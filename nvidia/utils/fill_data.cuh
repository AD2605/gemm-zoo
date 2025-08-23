#ifndef NVIDIA_UTILS_FILL_DATA_CUH
#define NVIDIA_UTILS_FILL_DATA_CUH

#include "defines.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdint>

namespace utils {

__global__ void setup_curand_states(curandState *states,
                                    unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed + idx, idx, 0, &states[idx]);
}

template <typename T>
__global__ void populate_with_random_kernel(T *device_ptr,
                                            curandState *d_states,
                                            std::size_t num_elements) {
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_elements) return;
  auto local_state = d_states[idx];

  for (; idx < num_elements; idx += blockDim.x * gridDim.x) {
    float rnd = curand_normal(&local_state) * 4 - 2;
    device_ptr[idx] = rnd;
  }
}

template <typename T>
void populate_with_random(T *device_ptr, std::size_t num_elements,
                          cudaStream_t stream) {
  dim3 block_dim(32, 1, 1);
  dim3 grid_dim((num_elements + 32 - 1) / 32, 1, 1);

  curandState *d_states;
  cudaMalloc(&d_states, num_elements * sizeof(curandState));

  setup_curand_states<<<grid_dim, block_dim, 0, stream>>>(d_states, 2025);
  checkCudaError(cudaStreamSynchronize(stream));
  populate_with_random_kernel<<<grid_dim, block_dim, 0, stream>>>(
      device_ptr, d_states, num_elements);
  checkCudaError(cudaStreamSynchronize(stream));

  cudaFree(d_states);
}
}  // namespace utils
#endif
