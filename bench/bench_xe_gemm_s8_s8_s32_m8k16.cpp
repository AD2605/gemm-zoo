
#include "benchmark.hpp"

#include <cstdint>
#include <intel/xe_gemm_s8_s8_s32_m8k16.hpp>

#include <sycl/sycl.hpp>
#include <vector>

int main() {
  using InputType = int8_t;
  using OutputType = int32_t;

  sycl::queue queue;
  using reference_kernel = intel::gemm_s8_s8_s32_m8k16<InputType, OutputType>;
  benchmark::benchmark_sizes<reference_kernel, InputType, OutputType>(
      {{1024, 1024, 1024}, {2048, 2048, 2048}, {4096, 4096, 4096}}, queue);
}
