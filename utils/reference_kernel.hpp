#ifndef UTILS_REFERENCE_KERNEL_HPP
#define UTILS_REFERENCE_KERNEL_HPP

#include "defines.hpp"
#include "sycl/queue.hpp"

#include <cstddef>
#include <sycl/sycl.hpp>

namespace utils {
namespace detail {
template <typename TIn, typename TOut>
INLINE void reference_gemm(const TIn* a, const TIn* b, const TOut* c, TOut* d,
                           std::size_t m, std::size_t n, std::size_t k,
                           TOut alpha, TOut beta, const sycl::nd_item<1>& it) {
  auto global_id = it.get_global_id(0);
  const auto output_row = global_id / n;
  const auto output_col = global_id % n;

  TOut acc = (TOut)0.0F;

  for (int i = 0; i < k; i++) {
    acc += static_cast<TOut>(a[output_row * k + i]) *
           static_cast<TOut>(b[i * n + output_col]);
  }
  acc = alpha * acc + beta * c[output_row * n + output_col];
  d[output_row * n + output_col] = acc;
}
}  // namespace detail

template <typename TIn, typename TOut>
struct reference_kernel {
  reference_kernel(std::size_t m, std::size_t n, std::size_t k,
                   sycl::queue& queue)
      : m(m), n(n), k(k) {
    launch_range = sycl::nd_range<1>(static_cast<std::size_t>(m * n),
                                     static_cast<std::size_t>(32));
  }

  sycl::event operator()(const TIn* a, const TIn* b, const TOut* c, TOut* d,
                         TOut alpha, TOut beta, sycl::queue& queue) {
    auto m_copy = m;
    auto n_copy = n;
    auto k_copy = k;
    return queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(launch_range, [=](sycl::nd_item<1> it) {
        [[clang::always_inline]] detail::reference_gemm(
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

}  // namespace utils

#endif
