#ifndef INTEL_KERNELS_XE_GEMM_S8_S8_S32_M8K16_HPP
#define INTEL_KERNELS_XE_GEMM_S8_S8_S32_M8K16_HPP

#include "intel/builtins.hpp"
#include "intel/cacheopts.hpp"
#include "intel/spirv_mma.hpp"

#include "intel/defines.hpp"

#include <cstddef>
#include <cstdint>
#include <sys/types.h>
#include <utils/defines.hpp>

namespace intel {
namespace kernels {
INLINE void xe_gemm_s8_s8_s32_m8k16(const int8_t* a, const int8_t* b,
                                    const int32_t* c, int32_t* d, int m, int n,
                                    int k, int32_t alpha, int32_t beta,
                                    const sycl::nd_item<1>& it) {
  auto num_sgs_in_wg =
      it.get_local_range(0) / 16;  // as sub-group size will be 16;
  auto num_sgs_in_kernel = num_sgs_in_wg * it.get_group_range(0);
  auto sg = it.get_sub_group();
  auto sg_id = it.get_group(0) * num_sgs_in_wg + sg.get_group_id();

  constexpr int block_height = 8;
  constexpr int block_width = 32;
  constexpr int k_tile_size = 32;

  const int blocks_per_row = (n + block_width - 1) / block_width;
  const int blocks_per_col = (m + block_height - 1) / block_height;
  const int total_tiles = blocks_per_row * blocks_per_col;

  std::size_t a_matrix_width = (k - 1) * sizeof(int8_t);  // also equal to pitch
  std::size_t b_matrix_width = (n - 1) * sizeof(int8_t);

  std::size_t result_matrix_width = (n - 1) * sizeof(int32_t);

  intel::int8 acc_registers;

  for (; sg_id < total_tiles; sg_id += num_sgs_in_kernel) {
    auto h_coord = (sg_id / blocks_per_row) * block_height;
    auto w_coord = (sg_id % blocks_per_row) * block_width;

#pragma unroll(8)
    for (uint8_t i = 0; i < 8; i++) {
      acc_registers[i] = 0;
    }

    for (int i = 0; i < k; i += k_tile_size) {
      auto a_tile = __builtin_IB_subgroup_block_read_flat_u8_m8k32v1(
          (intptr_t)(a), a_matrix_width, m - 1, a_matrix_width,
          uint2{static_cast<uint>(i), static_cast<uint>(h_coord)});
      auto b_tile = __builtin_IB_subgroup_block_read_flat_transform_u8_m32k16v1(
          (intptr_t)(b), b_matrix_width, k - 1, b_matrix_width,
          uint2{static_cast<uint>(w_coord), static_cast<uint>(i)});

      acc_registers = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
          16, a_tile, b_tile, acc_registers,
          SPIRV_MMAOperands::SPIRV_MatrixAInt8 |
              SPIRV_MMAOperands::SPIRV_MatrixBInt8);
    }

    auto c_tile = __builtin_IB_subgroup_block_read_flat_u32_m8k16v1(
        (intptr_t)(c), result_matrix_width, m - 1, result_matrix_width,
        uint2{static_cast<uint>(w_coord), static_cast<uint>(h_coord)});
#pragma unroll(8)
    for (uint8_t i = 0; i < 8; i++) {
      acc_registers[i] = alpha * acc_registers[i] + beta * c_tile[i];
    }
    __builtin_IB_subgroup_block_write_flat_u32_m8k16v1(
        (intptr_t)(d), result_matrix_width, m - 1, result_matrix_width,
        uint2{static_cast<uint>(w_coord), static_cast<uint>(h_coord)},
        *reinterpret_cast<intel::uint8*>(&acc_registers));
  }
}
}  // namespace kernels
}  // namespace intel

#endif
