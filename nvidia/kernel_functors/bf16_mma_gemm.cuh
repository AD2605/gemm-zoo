#ifndef NVIDIA_KERNEL_FUNCTORS_BF16_GEMM_CUH
#define NVIDIA_KERNEL_FUNCTORS_BF16_GEMM_CUH

#include "defines.hpp"
#include "kernels/sm_80/bf16_mma_gemm.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>

namespace nvidia::kernel_functors {

template <typename TIn, typename TOut, int M, int N, int K>
struct bf16_mma_gemm {
  bf16_mma_gemm(std::size_t m, std::size_t n, std::size_t k,
                const cudaDeviceProp& properties)
      : m(m), n(n), k(k) {
    static_assert(std::is_same_v<TIn, bf16_t>);
    static_assert(std::is_same_v<TOut, float>);
    assert(m % M == 0);
    assert(n % N == 0);
    assert(k % K == 0);
    constexpr int WM = 32;
    constexpr int WN = 64;
    static_assert(M % 64 == 0);
    static_assert(N % 64 == 0);

    constexpr int num_warps = (M / WM) * (N / WN);
    constexpr int num_threads = num_warps * 32;
    constexpr int NumBuffers = 6;

    smem_size_required =
        NumBuffers * K * sizeof(TIn) * (M + N) + WN * num_warps * sizeof(TIn);
    assert(smem_size_required < properties.sharedMemPerMultiprocessor);

    checkCudaError(cudaFuncSetAttribute(
        nvidia::kernels::sm80::bf16_mma_gemm<M, N, K, WM, WN, NumBuffers,
                                             num_threads>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_required));
    auto num_sms = static_cast<std::size_t>(properties.multiProcessorCount);
    blockDim = dim3(num_threads, 1, 1);
    auto m_tiles_required = (m + M - 1) / M;
    auto n_tiles_required = (n + N - 1) / N;
    auto num_tiles_required =
        std::min(num_sms, m_tiles_required * n_tiles_required);
    gridDim = dim3(num_tiles_required, 1, 1);
  }

  void operator()(const TIn* a, const TIn* b, const TOut* c, TOut* d,
                  const TOut alpha, const TOut beta, cudaStream_t stream) {
    constexpr int WM = 32;
    constexpr int WN = 64;
    constexpr int NumBuffers = 6;
    constexpr int num_warps = (M / WM) * (N / WN);
    constexpr int num_threads = num_warps * 32;
    nvidia::kernels::sm80::bf16_mma_gemm<M, N, K, WM, WN, NumBuffers,
                                         num_threads>
        <<<gridDim, blockDim, smem_size_required, stream>>>(
            a, b, c, d, static_cast<int32_t>(m), static_cast<int32_t>(n),
            static_cast<int32_t>(k), alpha, beta);
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
