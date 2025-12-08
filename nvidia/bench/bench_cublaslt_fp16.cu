#include "kernel_functors/cublaslt_gemm.cuh"
#include "utils/benchmark.cuh"

#include <cuda_fp16.h>

int main() {
  float alpha = 1.0f;
  float beta = 1.0f;

  using kernel_functor_struct =
      nvidia::kernel_functors::cublasLt_gemm<__half, __half>;
  benchmark::benchmark<kernel_functor_struct, __half, __half>(
      {{128, 128, 128},
       {256, 256, 256},
       {512, 512, 512},
       {1024, 1024, 1024},
       {2048, 2048, 2048},
       {4096, 4096, 4096},
       {8192, 8192, 8192}},
      alpha, beta);
}
