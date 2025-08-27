#ifndef NVIDIA_KERNELS_RING_BUFFER_CUH
#define NVIDIA_KERNELS_RING_BUFFER_CUH

#include <cstdint>

#include <cuda_runtime.h>

namespace nvidia::kernels::ring_buffer {
template <int N, int BufferSize>
class smem_ring_buffer {
 public:
  __device__ __host__ __forceinline__ smem_ring_buffer(int32_t smem_ptr) {
    // buffer_size to be in bytes
    for (int i = 0; i < N; i++) {
      buffer_areas[i] = smem_ptr + i * BufferSize;
    }
  }

  __device__ __host__ __forceinline__ int32_t get_current_buffer() {
    return buffer_areas[head++];
    head = head % N;
  }

 private:
  int32_t buffer_areas[N];
  int head = 0;
};
}  // namespace nvidia::kernels::ring_buffer

#endif
