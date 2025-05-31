#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <cstddef>
#include <cstdint>

#include <oneapi/math.hpp>
#include <oneapi/math/rng/device.hpp>
#include <type_traits>

namespace test {
template <typename T>
sycl::event populate_with_random(T* device_ptr, std::size_t num_elements,
                                 sycl::queue& queue,
                                 const int32_t seed = 1234) {
  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(num_elements, [=](std::size_t element_id) {
      using namespace oneapi::math::rng::device;
      philox4x32x10<> engine(seed, element_id);
      constexpr T min_value =
          std::is_signed_v<T> ? static_cast<T>(-10.0F) : static_cast<T>(1.0F);
      uniform<T> distribution(min_value, static_cast<T>(10.0F));
      // something I picked up from cutlass, create random numbers with exact
      // zeros so that one needn't bother with threshold value during testing.
      // This eliminates rounding issues and fp errors completely.
      T output_value =
          static_cast<T>(static_cast<int32_t>(generate(distribution, engine)));
      device_ptr[element_id] = output_value;
    });
  });
}

template <typename T>
bool compare_results(const T* output, const T* reference,
                     const std::size_t& num_elements, sycl::queue& queue) {
  int* is_different = sycl::malloc_shared<int>(1, queue);
  *is_different = 0;
  queue
      .submit([&](sycl::handler& cgh) {
        cgh.parallel_for(num_elements, [=](std::size_t element_id) {
          auto output_value = output[element_id];
          auto ref_value = reference[element_id];
          if (output_value != ref_value) {
#ifdef DEBUG
            sycl::ext::oneapi::experimental::printf(
                "wrong output value at element_id: %lu, ref_value: %f, "
                "comupted_value: %f\n",
                element_id, static_cast<float>(ref_value),
                static_cast<float>(output_value));
#endif
            *is_different = 1;
          }
        });
      })
      .wait_and_throw();
  bool is_passed = (*is_different == 0);
  sycl::free(is_different, queue);
  return is_passed;
}

template <typename TIn, typename TOut>
void compute_reference(const TIn* a, const TIn* b, TIn* c, int m, int n, int k,
                       const TOut alpha, const TOut beta, sycl::queue& queue) {
  // TODO: pass a matrix config POD struct here, containing
  // layout, and leading dimensions etc...
  // For now, just assume row major and matrix width = lda
  auto no_trans = oneapi::math::transpose::nontrans;
  oneapi::math::blas::row_major::gemm(queue, no_trans, no_trans, m, n, k, alpha,
                                      a, k, b, n, beta, c, n);
  queue.wait_and_throw();
}

}  // namespace test

#endif
