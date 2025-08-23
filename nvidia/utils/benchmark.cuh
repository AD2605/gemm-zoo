#ifndef NVIDIA_UTILS_BENCHMARK_CUH
#define NVIDIA_UTILS_BENCHMARK_CUH

#include "compare.cuh"
#include "cublaslt_gemm.cuh"
#include "defines.hpp"
#include "fill_data.cuh"

#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace benchmark {
namespace detail {
template <class Kernel, typename TIn, typename TOut>
void benchmark_impl(std::size_t m, std::size_t n, std::size_t k, TOut alpha,
                    TOut beta, cudaStream_t stream, const int num_repititions) {
  const int device_id = 0;
  cudaDeviceProp properties;
  cudaError_t err = cudaGetDeviceProperties(&properties, device_id);
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }

  const std::size_t l2_cache_size = properties.l2CacheSize;
  const std::size_t problem_size =
      (m * k * sizeof(TIn) + k * n * sizeof(TIn) + m * n * sizeof(TOut));
  const std::size_t num_problems_in_l2 =
      std::ceil(l2_cache_size / problem_size);
  const std::size_t num_problems_required = num_problems_in_l2 + 1;
  const std::size_t min_repitions_required = std::max(
      static_cast<std::size_t>(num_repititions), num_problems_required);

  std::vector<TIn*> a_matrices(num_problems_required, nullptr);
  std::vector<TIn*> b_matrices(num_problems_required, nullptr);
  std::vector<TOut*> c_matrices(num_problems_required, nullptr);
  TOut* d_out;
  TOut* d_ref;

  for (int i = 0; i < num_problems_required; i++) {
    cudaMalloc(&a_matrices[i], m * k * sizeof(TIn));
    cudaMalloc(&b_matrices[i], n * k * sizeof(TIn));
    cudaMalloc(&c_matrices[i], m * n * sizeof(TOut));
    utils::populate_with_random(a_matrices[i], m * k, stream);
    utils::populate_with_random(b_matrices[i], k * n, stream);
    utils::populate_with_random(c_matrices[i], m * n, stream);
  }

  cudaMalloc(&d_out, m * n * sizeof(TOut));
  cudaMalloc(&d_ref, m * n * sizeof(TOut));

  Kernel kernel(m, n, k, properties);

  kernel(a_matrices[0], b_matrices[0], c_matrices[0], d_out, alpha, beta,
         stream);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaStreamSynchronize(stream));

  utils::cublaslt_gemm(a_matrices[0], b_matrices[0], c_matrices[0], d_ref, m, n,
                       k, alpha, beta);
  checkCudaError(cudaGetLastError());
  cudaStreamSynchronize(stream);

  utils::compare_results(d_out, d_ref, m * n, stream);

  cudaEvent_t start, stop;
  checkCudaError(cudaEventCreate(&start));
  checkCudaError(cudaEventCreate(&stop));

  float total_time = 0;
  for (auto i = 0; i < min_repitions_required; i++) {
    float t = 0;
    int pointer_to_use = (i + 1) % a_matrices.size();
    checkCudaError(cudaEventRecord(start, stream));
    kernel(a_matrices[pointer_to_use], b_matrices[pointer_to_use],
           c_matrices[pointer_to_use], d_out, alpha, beta, stream);
    checkCudaError(cudaEventRecord(stop, stream));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&t, start, stop));
    total_time += t;
  }

  float average_time_in_seconds = (total_time / min_repitions_required) * 1e-3;
  const std::size_t flops_per_problem =
      static_cast<std::size_t>(m) * n * (2 * k + 3);

  float achieved_flops =
      (static_cast<float>(flops_per_problem) * 1e-9) / average_time_in_seconds;

  printf("GEMM Problem: [m: %lu n: %lu k: %lu], GFLOPS: %f: \n", m, n, k,
         achieved_flops);

  cudaFree(d_out);
  cudaFree(d_ref);

  for (int i = 0; i < a_matrices.size(); i++) {
    cudaFree(a_matrices[i]);
    cudaFree(b_matrices[i]);
    cudaFree(c_matrices[i]);
  }
}
}  // namespace detail

template <class Kernel, typename TIn, typename TOut>
void benchmark(std::vector<std::tuple<int, int, int>> problem_sizes, TOut alpha,
               TOut beta, const int num_repititions = 50) {
  cudaStream_t stream;
  cudaError_t err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to create CUDA stream:");
  }

  for (const auto& size_tuple : problem_sizes) {
    auto [m, n, k] = size_tuple;
    detail::benchmark_impl<Kernel, TIn, TOut>(m, n, k, alpha, beta, stream,
                                              num_repititions);
  }
  cudaStreamDestroy(stream);
}

}  // namespace benchmark

#endif
