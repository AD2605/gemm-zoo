#ifndef NVIDIA_KERNELS_MORTON_ENCODING_CUH
#define NVIDIA_KERNELS_MORTON_ENCODING_CUH

#include <cstdint>

namespace nvidia::kernels::utils {

__host__ __device__ __forceinline__ uint32_t morton_encode(uint32_t x,
                                                           uint32_t y) {
  x = (x | (x << 8u)) & 0x00FF00FFu;
  x = (x | (x << 4u)) & 0x0F0F0F0Fu;
  x = (x | (x << 2u)) & 0x33333333u;
  x = (x | (x << 1u)) & 0x55555555u;

  y = (y | (y << 8u)) & 0x00FF00FFu;
  y = (y | (y << 4u)) & 0x0F0F0F0Fu;
  y = (y | (y << 2u)) & 0x33333333u;
  y = (y | (y << 1u)) & 0x55555555u;

  return x | (y << 1u);
}

__host__ __device__ __forceinline__ void morton_decode(uint32_t code,
                                                       uint32_t& x,
                                                       uint32_t& y) {
  x = (code & 0xAAAAAAAAu) >> 1u;
  y = (code & 0x55555555u);

  x = (x | (x >> 1u)) & 0x33333333u;
  x = (x | (x >> 2u)) & 0x0F0F0F0Fu;
  x = (x | (x >> 4u)) & 0x00FF00FFu;
  x = (x | (x >> 8u)) & 0x0000FFFFu;

  y = (y | (y >> 1u)) & 0x33333333u;
  y = (y | (y >> 2u)) & 0x0F0F0F0Fu;
  y = (y | (y >> 4u)) & 0x00FF00FFu;
  y = (y | (y >> 8u)) & 0x0000FFFFu;
}
}  // namespace nvidia::kernels::utils

#endif
