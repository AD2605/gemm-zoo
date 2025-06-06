#ifndef INTEL_DEFINES_HPP
#define INTEL_DEFINES_HPP

#include <sys/types.h>

namespace intel {

#define SYCL_UNREACHABLE(x) \
  assert(0 && x);           \
  printf(x);

#ifdef __SYCL_DEVICE_ONLY__
template <class T, int N>
using vector_t = T __attribute__((ext_vector_type(N)));
#else
template <class T, int N>
using vector_t = sycl::marray<T, N>;
#endif

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_DEVICE_BUILTIN(x) SYCL_EXTERNAL extern "C" x
#else
#define SYCL_DEVICE_BUILTIN(x)                                             \
  inline x {                                                               \
    SYCL_UNREACHABLE("Attempting to use a device built-in in host code."); \
  }
#endif

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_DEVICE_OCL(x) SYCL_EXTERNAL extern "C" x
#else
#define SYCL_DEVICE_OCL(x)
#endif

using uint8 = vector_t<uint, 8>;
using uint4 = vector_t<uint, 4>;
using uint2 = vector_t<uint, 2>;

using short16 = vector_t<short, 16>;
using short8 = vector_t<short, 8>;
using short4 = vector_t<short, 4>;

using int8 = vector_t<int, 8>;

using float8 = vector_t<float, 8>;

}  // namespace intel

#endif
