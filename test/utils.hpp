#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_common.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include "oneapi/dnnl/dnnl_types.h"
#include <cstddef>
#include <cstdint>
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>

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
      const float min_value = std::is_signed_v<T> ? -10.0F : 1.0F;
      uniform<float> distribution(min_value, 10.0F);
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

template <typename T>
dnnl::memory::data_type dnnl_memory_data_type() {
  if constexpr (std::is_same_v<T, float>) {
    return dnnl::memory::data_type::f32;
  }
  if constexpr (std::is_same_v<T, sycl::half>) {
    return dnnl::memory::data_type::f16;
  }
  if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
    return dnnl::memory::data_type::bf16;
  }
  if constexpr (std::is_same_v<T, int32_t>) {
    return dnnl::memory::data_type::s32;
  }
  if constexpr (std::is_same_v<T, int8_t>) {
    return dnnl::memory::data_type::s8;
  }
  if constexpr (std::is_same_v<T, uint8_t>) {
    return dnnl::memory::data_type::u8;
  }
}

template <typename TIn, typename TOut>
void compute_reference(const TIn* a, const TIn* b, TOut* c, int m, int n, int k,
                       const TOut alpha, const TOut beta, sycl::queue& queue) {
  // using oneDNN here because oneMath unfortunately does not
  // provide different data type combinations.

  auto device = queue.get_device();
  auto context = queue.get_context();
  dnnl::engine engine = dnnl::sycl_interop::make_engine(device, context);
  dnnl::stream stream = dnnl::sycl_interop::make_stream(engine, queue);

  dnnl::memory::desc a_desc = dnnl::memory::desc(
      {m, k}, dnnl_memory_data_type<TIn>(), dnnl::memory::format_tag::ab);
  dnnl::memory::desc b_desc = dnnl::memory::desc(
      {k, n}, dnnl_memory_data_type<TIn>(), dnnl::memory::format_tag::ab);
  dnnl::memory::desc c_desc = dnnl::memory::desc(
      {m, n}, dnnl_memory_data_type<TOut>(), dnnl::memory::format_tag::ab);

  dnnl::primitive_attr matmul_attributes;
  matmul_attributes.set_scratchpad_mode(dnnl::scratchpad_mode::library);
  matmul_attributes.set_fpmath_mode(dnnl::fpmath_mode::any);
  matmul_attributes.set_scales_mask(DNNL_ARG_WEIGHTS, 0);
  dnnl::post_ops post_ops;
  post_ops.append_sum(beta, 0, dnnl_memory_data_type<TOut>());
  matmul_attributes.set_post_ops(post_ops);

  dnnl::matmul::primitive_desc matmul_pd(engine, a_desc, b_desc, c_desc,
                                         matmul_attributes);
  auto matmul_primitive = dnnl::matmul(matmul_pd);

  dnnl::memory a_memory(a_desc, engine, (void*)a);
  dnnl::memory b_memory(b_desc, engine, (void*)b);
  dnnl::memory c_memory(c_desc, engine, (void*)c);

  TOut* alpha_memory = sycl::malloc_device<TOut>(1, queue);
  queue.copy(&alpha, alpha_memory, 1).wait_and_throw();
  dnnl::memory::desc alpha_desc({1}, dnnl_memory_data_type<TOut>(),
                                dnnl::memory::format_tag::a);
  dnnl::memory alpha_memory_object(alpha_desc, engine, (void*)alpha_memory);

  matmul_primitive.execute(
      stream, {{DNNL_ARG_SRC, a_memory},
               {DNNL_ARG_WEIGHTS, b_memory},
               {DNNL_ARG_DST, c_memory},
               {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, alpha_memory_object}});
  stream.wait();
  queue.wait_and_throw();
}

}  // namespace test

#endif
