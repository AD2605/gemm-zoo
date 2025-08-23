#include "kernel_functors/naive_gemm.cuh"
#include "utils/benchmark.cuh"

#include <cuda_runtime.h>

#include <tuple>

int main() {
  float alpha = 1.0f;
  float beta = 1.0f;

  using kernel_functor_struct =
      nvidia::kernel_functors::naive_gemm<float, float>;
  benchmark::benchmark<kernel_functor_struct, float, float>(
      {{128, 128, 128}, {256, 256, 256}, {512, 512, 512}, {1024, 1024, 1024}},
      alpha, beta);
}
