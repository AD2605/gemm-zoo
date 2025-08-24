#ifndef NVIDIA_KERNEL_FUNCTORS_RMEM_TILED_GEMM
#define NVIDIA_KERNEL_FUNCTORS_RMEM_TILED_GEMM

#include "defines.hpp"
#include "kernels/rmem_tiled_gemm.cuh"

#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>

namespace nvidia::kernel_functors {
template <typename TIn, typename TOut, int M, int N, int K, int TM, int TN,
          int TK>
struct rmem_tiled_gemm {
  rmem_tiled_gemm(std::size_t m, std::size_t n, std::size_t k,
                  const cudaDeviceProp& properties)
      : m(m), n(n), k(k) {
    assert(m % M == 0);
    assert(n % N == 0);
    assert(k % K == 0);

    static_assert(TN % 4 == 0);
    static_assert(TK % 4 == 0);

    smem_size_required = K * sizeof(TIn) * (M + N);
    assert(smem_size_required < properties.sharedMemPerMultiprocessor);

    checkCudaError(cudaFuncSetAttribute(
        nvidia::kernels::rmem_tiled_gemm<TIn, TOut, M, N, K, TM, TN, TK>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_required));
    auto num_sms = static_cast<std::size_t>(properties.multiProcessorCount);
    blockDim = dim3(1024, 1, 1);
    auto m_tiles_required = (m + M - 1) / M;
    auto n_tiles_required = (n + N - 1) / N;
    auto num_tiles_required =
        std::min(num_sms, m_tiles_required * n_tiles_required);
    gridDim = dim3(num_tiles_required, 1, 1);
  }

  void operator()(const TIn* a, const TIn* b, const TOut* c, TOut* d,
                  const TOut alpha, const TOut beta, cudaStream_t stream) {
    nvidia::kernels::rmem_tiled_gemm<TIn, TOut, M, N, K, TM, TN, TK>
        <<<gridDim, blockDim, smem_size_required, stream>>>(a, b, c, d, m, n, k,
                                                            alpha, beta);
  }

 private:
  std::size_t m;
  std::size_t n;
  std::size_t k;
  dim3 blockDim;
  dim3 gridDim;
  std::size_t smem_size_required;
};
}  // namespace nvidia::kernel_functors

#endif
