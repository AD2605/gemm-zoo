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
      buffer_ptrs[i] = smem_ptr + i * BufferSize;
    }
  }

  __device__ __host__ __forceinline__ int32_t get_current_buffer() {
    return buffer_ptrs[head];
  }

  __device__ __host__ __forceinline__ int32_t get_tail_buffer() {
    return buffer_ptrs[tail];
  }

  __device__ __host__ __forceinline__ void pop() { head = (head + 1) % N; }

  __device__ __host__ __forceinline__ void push() { tail = (tail + 1) % N; }

 private:
  int32_t buffer_ptrs[N];
  int head = 0;
  int tail = 0;
};
}  // namespace nvidia::kernels::ring_buffer

#endif
