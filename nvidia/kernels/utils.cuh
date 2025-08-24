#ifndef NVIDIA_KERNELS_UTILS_CUH
#define NVIDIA_KERNELS_UTILS_CUH

namespace nvidia::kernels::utils {
__device__ __host__ __forceinline__ int ceil_div(int dividend, int divisor) {
  return (dividend + divisor - 1) / divisor;
}
}  // namespace nvidia::kernels::utils

#endif
