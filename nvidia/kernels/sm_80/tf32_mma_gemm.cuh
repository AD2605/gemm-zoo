#ifndef NVIDIA_KERNELS_SM80_TF32_MMA_GEMM_CUH
#define NVIDIA_KERNELS_SM80_TF32_MMA_GEMM_CUH

#include "kernels/hilbert.cuh"
#include "kernels/load_async.cuh"
#include "kernels/morton_encoding.cuh"
#include "kernels/ring_buffer.cuh"
#include "kernels/utils.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace nvidia::kernels::sm80 {

template <int BM, int BN, int BK, int WM, int WN, int NumBuffers,
          int NumThreads>
__launch_bounds__(NumThreads) __global__
    void tf32_mma_gemm(const float* a, const float* b, const float* c, float* d,
                       int32_t m, int32_t n, int32_t k, float alpha,
                       float beta) {
  constexpr int NumBanks = 32;
  constexpr int BankSize = 4;
  constexpr int elements_per_copy = 16 / sizeof(float);
  constexpr int M = utils::log2_floor(elements_per_copy);
  constexpr int B =
      utils::log2_floor((NumBanks * BankSize) / sizeof(float)) - M;
  constexpr int S_A = utils::log2_floor(BK) - M;
  constexpr int S_B = utils::log2_floor(BN) - M;
  static_assert(BK % 2 == 0);
  static_assert(BN % 128 == 0);
  static_assert(WN == 64);

  extern __shared__ char smem[];

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  constexpr int num_warps_per_row = BN / WN;
  auto warp_tile_row_id = warp_id / num_warps_per_row;
  auto warp_tile_col_id = warp_id % num_warps_per_row;

  constexpr int32_t sizeof_TIn = static_cast<int32_t>(sizeof(float));
  int32_t smem_a_addr = static_cast<int32_t>(__cvta_generic_to_shared(smem));
  int32_t smem_b_addr = smem_a_addr + BM * BK * sizeof_TIn * NumBuffers;
  int32_t smem_c_addr = smem_b_addr + BK * BN * sizeof_TIn * NumBuffers +
                        warp_id * WN * sizeof_TIn;

  int block_id = blockIdx.x;

  const int M_tiles = utils::ceil_div(m, BM);
  const int N_tiles = utils::ceil_div(n, BN);
  const int total_tiles = M_tiles * N_tiles;

  int group_id = lane_id >> 2;
  int thread_id_in_group = lane_id % 4;

  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 8;

  constexpr int TM = 4;
  constexpr int TN = 2;
  constexpr int TC = 4;

  const int num_matrix = lane_id / 8;
  const int num_matrix_row_offset = (num_matrix % 2) * 8 + lane_id % 8;
  const int num_matrix_col_offset = (num_matrix / 2) * 4;

  const int smem_load_a_offset =
      (warp_tile_row_id * WM + num_matrix_row_offset) * BK +
      num_matrix_col_offset;
  const int smem_load_b_offset = warp_tile_col_id * WN + group_id;

  uint32_t block_row;
  uint32_t block_col;

  utils::map_to_hilbert(block_id, N_tiles, block_col, block_row);

  int head = 0;
  int tail = 0;
  int k_load_index = 0;
#pragma unroll
  for (int i = 0; i < NumBuffers; i++) {
    if (k_load_index < k) {
      async_load::load_swizzled<float, BM, BK, NumThreads>(
          a, smem_a_addr + tail * BM * BK * 4, k, block_row * BM, k_load_index);
      async_load::load_swizzled<float, BK, BN, NumThreads>(
          b, smem_b_addr + tail * BK * BN * 4, n, k_load_index, block_col * BN);

      asm volatile("cp.async.commit_group;\n");
      tail = (tail + 1) % NumBuffers;
      k_load_index += BK;
    }
  }

  while (block_id < total_tiles) {
    float C_regs[WM / MMA_M][WN / MMA_N][TC];  // 4 x 4 x 4
    head = 0;
    for (int _i = 0; _i < WM / MMA_M; _i++) {
      for (int _j = 0; _j < WN / MMA_N; _j++) {
        for (int _k = 0; _k < TC; _k++) {
          C_regs[_i][_j][_k] = 0;
        }
      }
    }

    for (int kk = 0; kk < k; kk += BK) {
      asm volatile("cp.async.wait_group %0; \n" ::"n"(NumBuffers - 1));
      __syncthreads();

      int32_t smem_a_head_addr = smem_a_addr + head * BM * BK * 4;
      int32_t smem_b_head_addr = smem_b_addr + head * BK * BN * 4;
      // m16n8k8
      uint32_t A_regs[WM / MMA_M][2][TM];  // 2 x 4 x 4
      uint32_t B_regs[WN / MMA_N][2][TN];  // 2 x 4 x 2

#pragma unroll
      for (int _k = 0; _k < BK; _k += MMA_K) {
        int b_row0 = _k + thread_id_in_group;
        int b_row1 = b_row0 + 4;
#pragma unroll
        for (int _n = 0; _n < WN / MMA_N; _n++) {
          int b_col = smem_load_b_offset + _n * MMA_N;
          const int index_0 = utils::swizzle<B, M, S_B>(b_row0 * BN + b_col);
          const int index_1 = utils::swizzle<B, M, S_B>(b_row1 * BN + b_col);
          asm volatile("ld.shared.b32 %0, [%1]; \n"
                       : "=r"(B_regs[_n][_k % 2][0])
                       : "r"(smem_b_head_addr + index_0 * 4));
          asm volatile("ld.shared.b32 %0, [%1]; \n"
                       : "=r"(B_regs[_n][_k % 2][1])
                       : "r"(smem_b_head_addr + index_1 * 4));
        }
#pragma unroll
        for (int _m = 0; _m < WM / MMA_M; _m++) {
          int smem_a_index = smem_load_a_offset + _m * MMA_M * BK + _k;
          const int swizzled_address =
              utils::swizzle<B, M, S_A>(smem_a_index) * 4;
          asm volatile(
              "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, "
              "%3}, [%4];\n"
              : "=r"(A_regs[_m][_k % 2][0]), "=r"(A_regs[_m][_k % 2][1]),
                "=r"(A_regs[_m][_k % 2][2]), "=r"(A_regs[_m][_k % 2][3])
              : "r"(smem_a_head_addr + swizzled_address));
        }
#pragma unroll
        for (int _m = 0; _m < WM / MMA_M; _m++) {
          for (int _n = 0; _n < WN / MMA_N; _n++) {
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%0, %1, %2, %3}; \n"
                : "+f"(C_regs[_m][_n][0]), "+f"(C_regs[_m][_n][1]),
                  "+f"(C_regs[_m][_n][2]), "+f"(C_regs[_m][_n][3])
                : "r"(A_regs[_m][_k % 2][0]), "r"(A_regs[_m][_k % 2][1]),
                  "r"(A_regs[_m][_k % 2][2]), "r"(A_regs[_m][_k % 2][3]),
                  "r"(B_regs[_n][_k % 2][0]), "r"(B_regs[_n][_k % 2][1]));
          }
        }
      }

      head = (head + 1) % NumBuffers;

      if (k_load_index < k) {
        async_load::load_swizzled<float, BM, BK, NumThreads>(
            a, smem_a_addr + tail * BM * BK * 4, k, block_row * BM,
            k_load_index);
        async_load::load_swizzled<float, BK, BN, NumThreads>(
            b, smem_b_addr + tail * BK * BN * 4, n, k_load_index,
            block_col * BN);

        asm volatile("cp.async.commit_group;\n");
        tail = (tail + 1) % NumBuffers;
        k_load_index += BK;
      }
    }

    constexpr int ChunkLoads = (WM / 2);
    float bias_regs[ChunkLoads][2];
    const auto base_output_address =
        (static_cast<int32_t>(block_row) * BM + warp_tile_row_id * WM) * n +
        block_col * BN + warp_tile_col_id * WN + 2 * lane_id;
    const auto base_c_address = c + base_output_address;
    const auto base_d_address = d + base_output_address;

#pragma unroll
    for (int i = 0; i < ChunkLoads; i++) {
      asm volatile("ld.global.cs.v2.f32 {%0, %1}, [%2]; \n\t"
                   : "=f"(bias_regs[i][0]), "=f"(bias_regs[i][1])
                   : "l"(base_c_address + i * n));
    }

    block_id += gridDim.x;
    if (block_id < total_tiles) {
      tail = 0;
      k_load_index = 0;
      utils::map_to_hilbert(block_id, N_tiles, block_col, block_row);

#pragma unroll
      for (int i = 0; i < NumBuffers; i++) {
        if (k_load_index < k) {
          async_load::load_swizzled<float, BM, BK, NumThreads>(
              a, smem_a_addr + tail * BM * BK * 4, k, block_row * BM,
              k_load_index);
          async_load::load_swizzled<float, BK, BN, NumThreads>(
              b, smem_b_addr + tail * BK * BN * 4, n, k_load_index,
              block_col * BN);

          asm volatile("cp.async.commit_group;\n");
          tail = (tail + 1) % NumBuffers;
          k_load_index += BK;
        }
      }
    }

#pragma unroll
    for (int i = 0; i < WM; i++) {
      int responsible_thread_id_start = ((i % 8)) * 4;
      int responsible_thread_id_end = responsible_thread_id_start + 4;
      int reg_number = (i % 16) / 8;
      if (lane_id >= responsible_thread_id_start &&
          lane_id < responsible_thread_id_end) {
#pragma unroll
        for (int j = 0; j < WN; j += MMA_N) {
          int byte_offset = j * static_cast<int32_t>(sizeof(float));
          int smem_offset = (lane_id - responsible_thread_id_start) * 2 *
                                static_cast<int32_t>(sizeof(float)) +
                            byte_offset;
          asm volatile("st.shared.v2.f32 [%0], {%1, %2};\n" ::"r"(smem_c_addr +
                                                                  smem_offset),
                       "f"(C_regs[i / MMA_M][j / MMA_N][reg_number * 2]),
                       "f"(C_regs[i / MMA_M][j / MMA_N][reg_number * 2 + 1])
                       : "memory");
        }
      }
      __syncwarp();
      float result0, result1;
      asm volatile("ld.shared.v2.f32 {%0, %1}, [%2]; \n"
                   : "=f"(result0), "=f"(result1)
                   : "r"(smem_c_addr +
                         2 * lane_id * static_cast<int32_t>(sizeof(float))));
      bias_regs[i % ChunkLoads][0] *= beta;
      bias_regs[i % ChunkLoads][1] *= beta;
      result0 = alpha * result0 + bias_regs[i % ChunkLoads][0];
      result1 = alpha * result1 + bias_regs[i % ChunkLoads][1];
      asm volatile("st.global.cs.v2.f32 [%0], {%1, %2}; \n\t" ::"l"(
                       base_d_address + i * n),
                   "f"(result0), "f"(result1)
                   : "memory");
      // Pipeline the next load. Compiler Better remove the branch after
      // complete unrolling this loop
      if (i + ChunkLoads < WM) {
        asm volatile("ld.global.cs.v2.b32 {%0, %1}, [%2]; \n"
                     : "=f"(bias_regs[(i + ChunkLoads) % ChunkLoads][0]),
                       "=f"(bias_regs[(i + ChunkLoads) % ChunkLoads][1])
                     : "l"(base_c_address + (i + ChunkLoads) * n));
      }
    }
  }
}
}  // namespace nvidia::kernels::sm80

#endif
