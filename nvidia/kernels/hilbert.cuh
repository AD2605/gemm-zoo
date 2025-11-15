#ifndef NVIDIA_KERNELS_HILBERT_CUH
#define NVIDIA_KERNELS_HILBERT_CUH

namespace nvidia::kernels::utils {

__host__ __device__ __forceinline__ void rot(unsigned int n, unsigned int& x,
                                             unsigned int& y, unsigned int rx,
                                             unsigned int ry) {
  if (ry == 0) {
    if (rx == 1) {
      x = n - 1 - x;
      y = n - 1 - y;
    }
    unsigned int t = x;
    x = y;
    y = t;
  }
}

__host__ __device__ __forceinline__ void map_to_hilbert(
    const unsigned int linear_index, const unsigned int grid_side,
    unsigned int& block_x, unsigned int& block_y) {
  block_x = 0;
  block_y = 0;
  unsigned int t = linear_index;
  unsigned int s = 1;
  while (s < grid_side) {
    unsigned int rx = 1 & (t / 2);
    unsigned int ry = 1 & (t ^ rx);
    rot(s, block_x, block_y, rx, ry);
    block_x += s * rx;
    block_y += s * ry;
    t /= 4;
    s *= 2;
  }
}
}  // namespace nvidia::kernels::utils

#endif
