
#include "benchmark.hpp"
#include "sycl/ext/oneapi/bfloat16.hpp"

#include <intel/xe_gemm_bf16_bf16_f32_m8k16.hpp>

#include <sycl/sycl.hpp>
#include <vector>

int main() {
  using InputType = sycl::ext::oneapi::bfloat16;
  using OutputType = float;

  sycl::queue queue;
  using reference_kernel =
      intel::gemm_bf16_bf16_f32_m8k16<InputType, OutputType>;
  benchmark::benchmark_sizes<reference_kernel, InputType, OutputType>(
      {{1024, 1024, 1024}, {2048, 2048, 2048}, {4096, 4096, 4096}}, queue);
}
