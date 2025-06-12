#ifndef BENCH_BENCHMARK_HPP
#define BENCH_BENCHMARK_HPP

#include <cstddef>
#include <cstdio>
#include <memory>
#include <ratio>
#include <stdexcept>
#include <sycl/sycl.hpp>

#include <chrono>
#include <iostream>
#include <memory>

#include <tuple>
#include <vector>

#include <test/utils.hpp>

namespace benchmark {

namespace detail {

template <typename Kernel, typename TIn, typename TOut>
void benchmark_kernel(std::size_t m, std::size_t n, std::size_t k, TOut alpha,
                      TOut beta, const int num_repititions,
                      sycl::queue& queue) {
  using InputSharedPtr = std::shared_ptr<TIn>;
  using OutputSharedPtr = std::shared_ptr<TOut>;
  auto input_memory_deleter = [queue](TIn* ptr) { sycl::free(ptr, queue); };
  auto output_memory_deleter = [queue](TOut* ptr) { sycl::free(ptr, queue); };

  const std::size_t llc_size =
      queue.get_device().get_info<sycl::info::device::global_mem_cache_size>();
  const std::size_t global_size =
      queue.get_device().get_info<sycl::info::device::global_mem_size>();
  const std::size_t problem_size =
      (m * k * sizeof(TIn) + k * n * sizeof(TIn) + m * n * sizeof(TOut));
  const std::size_t num_problems_in_llc = std::ceil(llc_size / problem_size);
  const std::size_t num_problems_required = num_problems_in_llc + 1;
  const std::size_t min_repitions_required = std::max(
      static_cast<std::size_t>(num_repititions), num_problems_required);
  if (num_problems_required * problem_size > global_size) {
    std::cout << "cannot allocate " << num_problems_required * problem_size
              << " bytes, not benchmarking this configuration" << std::endl;
  }
  std::vector<InputSharedPtr> a_pointers;
  std::vector<InputSharedPtr> b_pointers;
  std::vector<OutputSharedPtr> c_pointers;
  OutputSharedPtr output_d_pointer;  // Just use one to pointer for the result
  OutputSharedPtr output_d_ref_pointer;

  output_d_pointer = OutputSharedPtr(sycl::malloc_device<TOut>(m * n, queue),
                                     output_memory_deleter);
  output_d_ref_pointer = OutputSharedPtr(
      sycl::malloc_device<TOut>(m * n, queue), output_memory_deleter);
  for (std::size_t i = 0; i < min_repitions_required; i++) {
    a_pointers.emplace_back(std::move(InputSharedPtr(
        sycl::malloc_device<TIn>(m * k, queue), input_memory_deleter)));
    b_pointers.emplace_back(std::move(InputSharedPtr(
        sycl::malloc_device<TIn>(k * n, queue), input_memory_deleter)));
    c_pointers.emplace_back(std::move(OutputSharedPtr(
        sycl::malloc_device<TOut>(m * n, queue), output_memory_deleter)));
  }

  int32_t seed = 2025;
  for (std::size_t i = 0; i < min_repitions_required; i++) {
    test::populate_with_random(a_pointers[i].get(), m * k, queue, seed++);
    test::populate_with_random(b_pointers[i].get(), k * n, queue, seed++);
    test::populate_with_random(c_pointers[i].get(), m * n, queue, seed++);
  }
  queue.wait_and_throw();

  Kernel kernel(m, n, k, queue);
  // use the first input for testing;
  kernel(a_pointers[0].get(), b_pointers[0].get(), c_pointers[0].get(),
         output_d_pointer.get(), alpha, beta, queue)
      .wait_and_throw();
  test::compute_reference(a_pointers[0].get(), b_pointers[0].get(),
                          c_pointers[0].get(), output_d_ref_pointer.get(), m, n,
                          k, alpha, beta, queue);
  if (!test::compare_results(output_d_pointer.get(), output_d_ref_pointer.get(),
                             m * n, queue)) {
    throw std::runtime_error("verification failed");
  }

  std::size_t time = 0;

  for (std::size_t i = 0; i < min_repitions_required; i++) {
    auto pointer_index_to_use = (i + 1) % min_repitions_required;
    auto t1 = std::chrono::high_resolution_clock::now();
    kernel(a_pointers[pointer_index_to_use].get(),
           b_pointers[pointer_index_to_use].get(),
           c_pointers[pointer_index_to_use].get(), output_d_pointer.get(),
           alpha, beta, queue)
        .wait();
    auto t2 = std::chrono::high_resolution_clock::now();
    time +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
  }

  const double average_time =
      static_cast<double>(time) / static_cast<double>(min_repitions_required);
  const std::size_t memory_operations_per_problem =
      problem_size + m * n * sizeof(TOut);
  const std::size_t flops_per_problem =
      static_cast<std::size_t>(m) * n * (2 * k + 3);
  const double achieved_gbps =
      static_cast<double>(memory_operations_per_problem) / average_time;
  const double achieved_flops =
      static_cast<double>(flops_per_problem) / average_time;
  printf("GEMM Problem: [m: %lu n: %lu k: %lu], GFLOPS: %lf GBPS: %lf\n", m, n,
         k, achieved_flops, achieved_gbps);
}
}  // namespace detail

template <typename Kernel, typename TIn, typename TOut>
void benchmark_sizes(
    const std::vector<std::tuple<std::size_t, std::size_t, std::size_t>>& sizes,
    sycl::queue& queue, const int num_repititions = 50) {
  const TOut alpha = static_cast<TOut>(1.0f);
  const TOut beta = static_cast<TOut>(1.0f);

  for (const auto [m, n, k] : sizes) {
    detail::benchmark_kernel<Kernel, TIn, TOut>(m, n, k, alpha, beta,
                                                num_repititions, queue);
  }
}

}  // namespace benchmark
#endif
