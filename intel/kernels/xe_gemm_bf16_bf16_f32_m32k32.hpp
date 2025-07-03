#ifndef INTEL_KERNELS_GEMM_BF16_BF16_F32_M32K32_HPP
#define INTEL_KERNELS_GEMM_BF16_BF16_F32_M32K32_HPP

#include "intel/builtins.hpp"
#include "intel/cacheopts.hpp"
#include "intel/spirv_mma.hpp"

#include "intel/defines.hpp"

#include <cstddef>
#include <cstdint>
#include <sys/types.h>
#include <utils/defines.hpp>

#include <sycl/sycl.hpp>

namespace intel {
namespace kernels {
using bf16_t = sycl::ext::oneapi::bfloat16;
INLINE void gemm_bf16_bf16_f32_m32k32(const bf16_t* a, const bf16_t* b,
                                     const float* c, float* d, std::size_t m,
                                     std::size_t n, std::size_t k, float alpha,
                                     float beta, const sycl::nd_item<1>& it) {
#ifdef __SYCL_DEVICE_ONLY__
  // each sub-group will be be responsible for a block of 8x16 tile of C Matrix;
  auto num_sgs_in_wg =
      it.get_local_range(0) / 16;  // as sub-group size will be 16;
  auto num_sgs_in_kernel = num_sgs_in_wg * it.get_group_range(0);
  auto sg = it.get_sub_group();
  auto sg_id = it.get_group(0) * num_sgs_in_wg + sg.get_group_id();

  constexpr int block_height = 32;
  constexpr int block_width = 32;
  constexpr int k_tile_size = 32;

  const int blocks_per_row = (n + block_width - 1) / block_width;
  const int blocks_per_col = (m + block_height - 1) / block_height;
  const int total_tiles = blocks_per_row * blocks_per_col;

  std::size_t a_matrix_width = (k - 1) * sizeof(bf16_t);  // also equal to pitch
  std::size_t b_matrix_width = (n - 1) * sizeof(bf16_t);

  std::size_t result_matrix_width = (n - 1) * sizeof(float);

  intel::float8 accumulator[8];

  for (; sg_id < total_tiles; sg_id += num_sgs_in_kernel) {
    auto h_coord = (sg_id / blocks_per_row) * block_height;
    auto w_coord = (sg_id % blocks_per_row) * block_width;

    #pragma unroll(8)
    for (int8_t j = 0; j < 8; j++) {
      #pragma unroll(8)
      for (int8_t l = 0; l < 8; l++) {
        accumulator[j][l] = 0;
      }
    }

    for (int i = 0; i < k; i += k_tile_size) {
      intel::short64 a_tile = __builtin_IB_subgroup_block_read_flat_u16_m32k32v1(
          (intptr_t)(a), a_matrix_width, m - 1, a_matrix_width,
          uint2{static_cast<uint>(i), static_cast<uint>(h_coord)});
      intel::short64 b_tile =
          __builtin_IB_subgroup_block_read_flat_u16_m32k32v1(
              (intptr_t)(b), b_matrix_width, k - 1, b_matrix_width,
              uint2{static_cast<uint>(w_coord), static_cast<uint>(i)});

      
      
    }
  }
#endif
}
}  // namespace kernels
}  // namespace intel

#endif
