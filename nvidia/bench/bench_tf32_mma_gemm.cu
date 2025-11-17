#include "kernel_functors/tf32_mma_gemm.cuh"
#include "utils/benchmark.cuh"

#include <cuda_runtime.h>

#include <tuple>

int main() {
  constexpr int M = 128;
  constexpr int N = 128;
  constexpr int K = 32;

  float alpha = 1.0f;
  float beta = 1.0f;

  using kernel_functor_struct =
      nvidia::kernel_functors::tf32_mma_gemm<float, float, M, N, K>;
  benchmark::benchmark<kernel_functor_struct, float, float>(
      {{1024, 1024, 1024},
       {2048, 2048, 2048},
       {4096, 4096, 4096},
       {8192, 8192, 8192}},
      alpha, beta);
}
