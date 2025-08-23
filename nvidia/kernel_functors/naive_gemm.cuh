#ifndef NVIDIA_KERNEL_FUNCTORS_NAIVE_GEMM_HPP
#define NVIDIA_KERNEL_FUNCTORS_NAIVE_GEMM_HPP

#include "kernels/naive_gemm.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace nvidia::kernel_functors {
template <typename TIn, typename TOut>
struct naive_gemm {
  naive_gemm(std::size_t m, std::size_t n, std::size_t k, const cudaDeviceProp&)
      : m(m), n(n), k(k) {
    blockDim = dim3(32, 32, 1);
    gridDim = dim3((n + 32 - 1) / 32, (m + 32 - 1) / 32, 1);
  }

  void operator()(const TIn* a, const TIn* b, const TOut* c, TOut* d,
                  const TOut alpha, const TOut beta, cudaStream_t stream) {
    nvidia::kernels::naive_gemm<TIn, TOut>
        <<<gridDim, blockDim, 0, stream>>>(a, b, c, d, m, n, k, alpha, beta);
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
