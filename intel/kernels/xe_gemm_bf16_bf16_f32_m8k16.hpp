#ifndef INTEL_KERNELS_GEMM_BF16_BF16_F32_M8k16_HPP
#define INTEL_KERNELS_GEMM_BF16_BF16_F32_M8k16_HPP

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
INLINE void gemm_bf16_bf16_f32_m8k16(const bf16_t* a, const bf16_t* b,
                                     const float* c, float* d, std::size_t m,
                                     std::size_t n, std::size_t k, float alpha,
                                     float beta, const sycl::nd_item<1>& it) {
#ifdef __SYCL_DEVICE_ONLY__
  using float8 = sycl::vec<float, 8>;
  // each sub-group will be be responsible for a block of 8x16 tile of C Matrix;
  auto num_sgs_in_wg =
      it.get_local_range(0) / 16;  // as sub-group size will be 16;
  auto num_sgs_in_kernel = num_sgs_in_wg * it.get_group_range(0);
  auto sg = it.get_sub_group();
  auto sg_id = it.get_group(0) * num_sgs_in_wg + sg.get_group_id();

  constexpr int block_height = 8;
  constexpr int block_width = 16;
  constexpr int k_tile_size = 32;

  const int blocks_per_row = (n + block_width - 1) / block_width;
  const int blocks_per_col = (m + block_height - 1) / block_height;
  const int total_tiles = blocks_per_row * blocks_per_col;

  std::size_t a_matrix_width = (k - 1) * sizeof(bf16_t);  // also equal to pitch
  std::size_t b_matrix_width = (n - 1) * sizeof(bf16_t);

  std::size_t result_matrix_width = (n - 1) * sizeof(float);

  intel::float8 acc_registers;
  for (; sg_id < total_tiles; sg_id += num_sgs_in_kernel) {
    auto h_coord = (sg_id / blocks_per_row) * block_height;
    auto w_coord = (sg_id % blocks_per_row) * block_width;

#pragma unroll(8)
    for (uint8_t i = 0; i < 8; i++) {
      acc_registers[i] = 0;
    }

    short8 a_tile_0;
    short8 a_tile_1;

    short16 b_tile_0;
    short16 b_tile_1;

    for (int i = 0; i < k; i += k_tile_size) {
      intel::short16 a_tile = __builtin_IB_subgroup_block_read_flat_u16_m8k32v1(
          (intptr_t)(a), a_matrix_width, m - 1, a_matrix_width,
          uint2{static_cast<uint>(i), static_cast<uint>(h_coord)});
      intel::short32 b_tile =
          __builtin_IB_subgroup_block_read_flat_u16_m32k16v1(
              (intptr_t)(b), b_matrix_width, k - 1, b_matrix_width,
              uint2{static_cast<uint>(w_coord), static_cast<uint>(i)});

      // prepare the two a tiles and b_tiles for dpas;
// I would expect all these index calculations to be constexpr and branchless
#pragma unroll(16)
      for (uint8_t i = 0; i < 16; i++) {
        if (i % 2 == 0) {
          a_tile_0[i / 2] = a_tile[i];
        } else {
          a_tile_1[i / 2] = a_tile[i];
        }
      }

#pragma unroll(32)
      for (uint8_t i = 0; i < 32; i++) {
        if (i / 16 == 0) {
          b_tile_0[i] = b_tile[i];
        } else {
          b_tile_1[i % 16] = b_tile[i];
        }
      }

      acc_registers = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
          16, a_tile_0, *reinterpret_cast<int8*>(&b_tile_0), acc_registers,
          SPIRV_MMAOperands::SPIRV_MatrixABf16 |
              SPIRV_MMAOperands::SPIRV_MatrixBBf16);
      acc_registers = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
          16, a_tile_1, *reinterpret_cast<int8*>(&b_tile_1), acc_registers,
          SPIRV_MMAOperands::SPIRV_MatrixABf16 |
              SPIRV_MMAOperands::SPIRV_MatrixBBf16);
    }

    auto c_tile_uint = __builtin_IB_subgroup_block_read_flat_u32_m8k16v1(
        (intptr_t)(c), result_matrix_width, m - 1, result_matrix_width,
        uint2{static_cast<uint>(w_coord), static_cast<uint>(h_coord)});
    float8 c_tile = *reinterpret_cast<float8*>(&c_tile_uint);
#pragma unroll(8)
    for (uint8_t i = 0; i < 8; i++) {
      acc_registers[i] = alpha * acc_registers[i] + beta * c_tile[i];
    }
    __builtin_IB_subgroup_block_write_flat_u32_m8k16v1(
        (intptr_t)(d), result_matrix_width, m - 1, result_matrix_width,
        uint2{static_cast<uint>(w_coord), static_cast<uint>(h_coord)},
        *reinterpret_cast<intel::uint8*>(&acc_registers));
  }
#endif
}
}  // namespace kernels
}  // namespace intel

#endif
