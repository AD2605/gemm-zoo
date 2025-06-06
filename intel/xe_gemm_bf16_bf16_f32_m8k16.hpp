#ifndef INTEL_GEMM_BF16_BF16_F32_M8k16_HPP
#define INTEL_GEMM_BF16_BF16_F32_M8k16_HPP

#include "intel/kernels/xe_gemm_bf16_bf16_f32_m8k16.hpp"
#include "sycl/event.hpp"
#include <cstddef>
#include <sycl/sycl.hpp>

namespace intel {

template <typename TIn, typename TOut>
struct gemm_bf16_bf16_f32_m8k16 {
  gemm_bf16_bf16_f32_m8k16(std::size_t m, std::size_t n, std::size_t k,
                           sycl::queue& queue)
      : m(m), n(n), k(k) {
    constexpr int block_height = 8;
    constexpr int block_width = 16;
    std::size_t num_sub_groups_required =
        ((m + block_height - 1) / block_height) *
        ((n + block_width - 1) / block_width);
    std::size_t local_range = 16;
    std::size_t global_range = num_sub_groups_required * local_range;
    // Tune it later.
    launch_range = sycl::nd_range<1>({global_range}, {local_range});
  }

  sycl::event operator()(const TIn* a, const TIn* b, const TOut* c, TOut* d,
                         TOut alpha, TOut beta, sycl::queue& queue) {
    auto m_copy = m;
    auto n_copy = n;
    auto k_copy = k;
    return queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          launch_range,
          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            [[clang::always_inline]] intel::kernels::gemm_bf16_bf16_f32_m8k16(
                a, b, c, d, m_copy, n_copy, k_copy, alpha, beta, it);
          });
    });
  }

 private:
  std::size_t m;
  std::size_t n;
  std::size_t k;
  sycl::nd_range<1> launch_range;
};
}  // namespace intel

#endif
