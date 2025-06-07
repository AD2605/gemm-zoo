#ifndef INTEL_KERNELS_GEMM_BF16_BF16_F32_M8k16_HPP
#define INTEL_KERNELS_GEMM_BF16_BF16_F32_M8k16_HPP

#include "intel/builtins.hpp"
#include "intel/cacheopts.hpp"
#include "intel/spirv_mma.hpp"

#include "intel/defines.hpp"
#include "sycl/ext/oneapi/bfloat16.hpp"
#include "sycl/nd_item.hpp"
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
  constexpr int k_tile_size = 16;

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

    // begin prefetch of A and B tile, Then load C tile and multiply it with
    // Beta.
    // I fetch the C tile before starting the matmuls
    // so that it does not stall later when doing the
    // addition with the result. Since GPU execution model
    // is "fire-and-forget", I expect the GPU to schedule
    // this load and move ahead, and since there is no dependency
    // on this load until the end, the result of load should be ready
    // in the meantine when we need it.

    // TODO: Commenting out prefetches because they give much better
    // performance. Need to understand why

    /*__builtin_IB_subgroup_block_read_prefetch_u16_m8k16v1(
        (intptr_t)(a), a_matrix_width, m - 1, a_matrix_width,
        uint2{0, static_cast<uint>(h_coord)},
        intel::cacheopts::LSC_LDCC_L1C_L3C);
    __builtin_IB_subgroup_block_read_prefetch_transform_u16_k16(
        (intptr_t)(b), b_matrix_width, k - 1, b_matrix_width,
        intel::uint2{static_cast<uint>(w_coord), 0},
        intel::cacheopts::LSC_LDCC_L1C_L3C);
    */
    auto c_tile_uint = __builtin_IB_subgroup_block_read_flat_u32_m8k16v1(
        (intptr_t)(c), result_matrix_width, m - 1, result_matrix_width,
        uint2{static_cast<uint>(w_coord), static_cast<uint>(h_coord)});
    float8 c_tile = *reinterpret_cast<float8*>(&c_tile_uint);
#pragma unroll(8)
    for (uint8_t i = 0; i < 8; i++) {
      acc_registers[i] = 0;
    }
    for (int i = 0; i < k; i += k_tile_size) {
      intel::short8 a_tile = __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(
          (intptr_t)(a), a_matrix_width, m - 1, a_matrix_width,
          uint2{static_cast<uint>(i), static_cast<uint>(h_coord)});
      intel::int8 b_tile =
          __builtin_IB_subgroup_block_read_flat_transform_u16_k16(
              (intptr_t)(b), b_matrix_width, k - 1, b_matrix_width,
              uint2{static_cast<uint>(w_coord), static_cast<uint>(i)});

      // These prefetches are the killer
      /*if ((i + k_tile_size) < k) {
        __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v1(
            (intptr_t)(a), a_matrix_width, m - 1, a_matrix_width,
            uint2{static_cast<uint>(i + k_tile_size),
                  static_cast<uint>(h_coord)},
            intel::cacheopts::LSC_LDCC_L1C_L3C);
        __builtin_IB_subgroup_block_read_prefetch_transform_u16_k16(
            (intptr_t)(b), b_matrix_width, k - 1, b_matrix_width,
            uint2{static_cast<uint>(w_coord),
                  static_cast<uint>(i + k_tile_size)},
            intel::cacheopts::LSC_LDCC_L1C_L3C);
      }*/

      acc_registers = __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
          16, a_tile, b_tile, acc_registers,
          SPIRV_MMAOperands::SPIRV_MatrixABf16 |
              SPIRV_MMAOperands::SPIRV_MatrixBBf16);
    }
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
