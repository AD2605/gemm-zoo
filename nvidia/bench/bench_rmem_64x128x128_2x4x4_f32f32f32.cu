#include "kernel_functors/rmem_tiled_gemm.cuh"
#include "utils/benchmark.cuh"

#include <cuda_runtime.h>

#include <tuple>

int main() {
  constexpr int M = 128;
  constexpr int N = 128;
  constexpr int K = 64;

  constexpr int TM = 16;
  constexpr int TN = 4;
  constexpr int TK = 4;

  float alpha = 1.0f;
  float beta = 1.0f;

  using kernel_functor_struct =
      nvidia::kernel_functors::rmem_tiled_gemm<float, float, M, N, K, TM, TN,
                                               TK>;
  benchmark::benchmark<kernel_functor_struct, float, float>(
      {{1024, 1024, 1024}, {2048, 2048, 2048}, {4096, 4096, 4096}}, alpha,
      beta);
}
