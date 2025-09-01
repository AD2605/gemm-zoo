#ifndef NVIDIA_KERNELS_NAIVE_GEMM_CUH
#define NVIDIA_KERNELS_NAIVE_GEMM_CUH

#include <cute/tensor.hpp>

namespace nvidia::kernels {
template <typename TIn, typename TOut>
__global__ void naive_gemm(const TIn* a, const TIn* b, const TOut* c, TOut* d,
                           std::size_t m, std::size_t n, std::size_t k,
                           TOut alpha, TOut beta) {
  using namespace cute;
  Tensor gmem_tensor_a =
      make_tensor(make_gmem_ptr(a), make_layout(make_shape(m, k), make_stride(k, 1)));
  Tensor gmem_tensor_b =
      make_tensor(make_gmem_ptr(b), make_layout(make_shape(k, n), make_stride(n, 1)));
  Tensor gmem_tensor_c =
      make_tensor(make_gmem_ptr(c), make_layout(make_shape(m, n), make_stride(n, 1)));
  Tensor gmem_tensor_d =
      make_tensor(make_gmem_ptr(d), make_layout(make_shape(m, n), make_stride(n, 1)));

  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if (row >= m || col >= n) {
    return;
  }

  TOut acc = 0;
  for (int i = 0; i < k; i++) {
    auto a_coord = make_coord(row, i);
    auto b_coord = make_coord(i, col);

    acc += gmem_tensor_a[a_coord] * gmem_tensor_b[b_coord];
  }

  auto output_coord = make_coord(row, col);
  gmem_tensor_d[output_coord] =
      alpha * acc + beta * gmem_tensor_c[output_coord];
}
}  // namespace nvidia::kernels
#endif
