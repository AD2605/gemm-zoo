#ifndef NVIDIA_KERNELS_TILED_GEMM_CUH
#define NVIDIA_KERNELS_TILED_GEMM_CUH

#include "kernels/utils.cuh"

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/algorithm/copy.hpp>

#include <cstddef>
#include <cuda_runtime.h>

namespace nvidia::kernels {

template <typename TIn, typename TOut, int BM, int BN, int BK,
          int NumThreads = 128>
__launch_bounds__(NumThreads) __global__
    void smem_tiled_gemm(const TIn* a, const TIn* b, const TOut* c, TOut* d,
                         std::size_t m, std::size_t n, std::size_t k,
                         TOut alpha, TOut beta) {
  using namespace cute;
  
  extern __shared__ char smem[];
  
  Tensor gmem_tensor_a = make_tensor(
      make_gmem_ptr(a), make_layout(make_shape(m, k), make_stride(k, 1)));
  Tensor gmem_tensor_b = make_tensor(
      make_gmem_ptr(b), make_layout(make_shape(k, n), make_stride(n, 1)));
  Tensor gmem_tensor_c = make_tensor(
      make_gmem_ptr(c), make_layout(make_shape(m, n), make_stride(n, 1)));
  Tensor gmem_tensor_d = make_tensor(
      make_gmem_ptr(d), make_layout(make_shape(m, n), make_stride(n, 1)));

  const auto bM = Int<BM>{};
  const auto bN = Int<BN>{};
  const auto bK = Int<BK>{};

  // Tile shape, to be copied into Shared Memory
  auto cta_tiler_shape = make_shape(bM, bN, bK);
  Layout sA_layout = make_layout(make_shape(bM, bK), make_stride(bK, 1));
  Layout sB_layout = make_layout(make_shape(bK, bN), make_stride(bN, 1));

  Layout Thr_Layout_A = make_layout(make_shape(Int<NumThreads>{}),
                                    make_stride(Int<32>{}, Int<1>{}));

  using SMemCopyAtomA = UniversalCopy<4 * sizeof(TIn)>{};
  using SMemCopyAtomB = UniversalCopy<4 * sizeof(TIn)>{};

  TIn* smem_a_ptr = reinterpret_cast<TIn*>(smem);
  TIn* smem_b_ptr = smem_a_ptr + M * K * sizeof(TIn);

  
}
}  // namespace nvidia::kernels

#endif
