
#include "benchmark.hpp"

#include <utils/reference_kernel.hpp>

#include <sycl/sycl.hpp>
#include <vector>

int main() {
  using InputType = float;
  using OutputType = float;

  sycl::queue queue;
  using reference_kernel = utils::reference_kernel<InputType, OutputType>;
  benchmark::benchmark_sizes<reference_kernel, InputType, OutputType>(
      {{1024, 1024, 1024}, {2048, 2048, 2048}, {4096, 4096, 4096}}, queue);
}
