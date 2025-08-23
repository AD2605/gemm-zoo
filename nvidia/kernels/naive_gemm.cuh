#ifndef NAIVE_GEMM_CUH
#define NAIVE_GEMM_CUH

namespace nvidia::kernels {
template <typename TIn, typename TOut>
__global__ void naive_gemm(const TIn* a, const TIn* b, const TOut* c, TOut* d,
                           std::size_t m, std::size_t n, std::size_t k,
                           TOut alpha, TOut beta) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if (row >= m || col >= n) {
    return;
  }

  TOut acc = 0;
  for (int i = 0; i < k; i++) {
    acc += a[row * k + i] * b[i * n + col];
  }
  d[row * n + col] = alpha * acc + beta * c[row * n + col];
}
}  // namespace nvidia::kernels
#endif
