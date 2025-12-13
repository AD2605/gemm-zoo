#include "kernel_functors/bf16_mma_gemm.cuh"
#include "utils/benchmark.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <tuple>

int main() {
  using bf16_t = __nv_bfloat16;
  constexpr int M = 128;
  constexpr int N = 128;
  constexpr int K = 32;

  float alpha = 1.0f;
  float beta = 1.0f;

  using kernel_functor_struct =
      nvidia::kernel_functors::bf16_mma_gemm<bf16_t, float, M, N, K>;
  benchmark::benchmark<kernel_functor_struct, bf16_t, float>(
      {{256, 256, 256},
       {512, 512, 512},
       {1024, 1024, 1024},
       {2048, 2048, 2048},
       {4096, 4096, 4096},
       {8192, 8192, 8192}},
      alpha, beta);
}
