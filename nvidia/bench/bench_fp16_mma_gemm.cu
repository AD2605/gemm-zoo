#include "kernel_functors/fp16_mma_gemm.cuh"
#include "utils/benchmark.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <tuple>

int main() {
  constexpr int M = 256;
  constexpr int N = 256;
  constexpr int K = 32;

  __half alpha = __half(1.0f);
  __half beta = __half(1.0f);

  using kernel_functor_struct =
      nvidia::kernel_functors::fp16_mma_gemm<__half, __half, M, N, K>;
  benchmark::benchmark<kernel_functor_struct, __half, __half>(
      {{256, 256, 256},
       {512, 512, 512},
       {1024, 1024, 1024},
       {2048, 2048, 2048},
       {4096, 4096, 4096},
       {8192, 8192, 8192}},
      alpha, beta);
}
