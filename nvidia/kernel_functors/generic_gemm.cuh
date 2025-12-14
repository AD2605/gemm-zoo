#ifndef NVIDIA_KERNEL_FUNCTORS_GENERIC_GEMM_CUH
#define NVIDIA_KERNEL_FUNCTORS_GENERIC_GEMM_CUH

#include "defines.hpp"
#include "kernels/generic_gemm.cuh"

#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>

namespace nvidia::kernel_functors {
template <typename TIn, typename TOut>
struct generic_gemm {
  generic_gemm(std::size_t m, std::size_t n, std::size_t k) : m(m), n(n), k(k) {
    constexpr int M = 32;
    constexpr int N = 32;
    blockDim = dim3(32, 32, 1);
    auto m_tiles_required = (m + M - 1) / M;
    auto n_tiles_required = (n + N - 1) / N;
    auto num_tiles_required = m_tiles_required * n_tiles_required;
    gridDim = dim3(n_tiles_required, m_tiles_required, 1);
  }

  void operator()(const TIn* a, const TIn* b, const TOut* c, TOut* d,
                  const TOut alpha, const TOut beta, cudaStream_t stream) {
    int32_t m_int = static_cast<int32_t>(m);
    int32_t n_int = static_cast<int32_t>(n);
    int32_t k_int = static_cast<int32_t>(k);
    nvidia::kernels::generic_gemm<TIn, TOut><<<gridDim, blockDim, 0, stream>>>(
        a, b, c, d, alpha, beta, m_int, n_int, k_int, k_int, n_int, n_int,
        n_int);
  }

 private:
  std::size_t m;
  std::size_t n;
  std::size_t k;
  dim3 blockDim;
  dim3 gridDim;
};
}  // namespace nvidia::kernel_functors
#endif
