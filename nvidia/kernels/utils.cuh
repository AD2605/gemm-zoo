#ifndef NVIDIA_KERNELS_UTILS_CUH
#define NVIDIA_KERNELS_UTILS_CUH

#include <cstdint>

namespace nvidia::kernels::utils {
__device__ __host__ __forceinline__ int ceil_div(int dividend, int divisor) {
  return (dividend + divisor - 1) / divisor;
}

constexpr unsigned __device__ __forceinline__ log2_floor(int n) {
  unsigned r = 0;
  while (n >>= 1) ++r;
  return r;
}

template <int B, int M, int S>
__device__ __forceinline__ int swizzle(const int32_t &logical_offset) {
  uint32_t x = logical_offset;
  // Removed extraneous x ^= (x >> S); to match standard Cutlass/CuTe swizzle.
  x ^= ((x >> (M + S)) & ((1u << B) - 1)) << M;
  return static_cast<int32_t>(x);
}

}  // namespace nvidia::kernels::utils

#endif
