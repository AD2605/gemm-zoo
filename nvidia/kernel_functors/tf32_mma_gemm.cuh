#ifndef NVIDIA_KERNEL_FUNCTORS_TF32_MMA_GEMM_CUH
#define NVIDIA_KERNEL_FUNCTORS_TF32_MMA_GEMM_CUH

#include "defines.hpp"
#include "kernels/sm_80/tf32_mma_gemm.cuh"

#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>

namespace nvidia::kernel_functors {

template <typename TIn, typename TOut, int M, int N, int K>
struct tf32_mma_gemm {
  tf32_mma_gemm(std::size_t m, std::size_t n, std::size_t k,
                const cudaDeviceProp& properties)
      : m(m), n(n), k(k) {
    assert(m % M == 0);
    assert(n % N == 0);
    assert(k % K == 0);

    static_assert(M % (256 / 8) == 0);

    smem_size_required = 3 * K * sizeof(TIn) * (M + N);
    assert(smem_size_required < properties.sharedMemPerMultiprocessor);

    checkCudaError(cudaFuncSetAttribute(
        nvidia::kernels::sm80::tf32_mma_gemm<TIn, TOut, M, N, K, 128, 3>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_required));
    auto num_sms = static_cast<std::size_t>(properties.multiProcessorCount);
    blockDim = dim3(128, 1, 1);
    auto m_tiles_required = (m + M - 1) / M;
    auto n_tiles_required = (n + N - 1) / N;
    auto num_tiles_required =
        std::min(num_sms, m_tiles_required * n_tiles_required);
    gridDim = dim3(num_tiles_required, 1, 1);
  }

  void operator()(const TIn* a, const TIn* b, const TOut* c, TOut* d,
                  const TOut alpha, const TOut beta, cudaStream_t stream) {
    nvidia::kernels::sm80::tf32_mma_gemm<TIn, TOut, M, N, K, 128, 3>
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
