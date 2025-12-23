#ifndef NVIDIA_KERNELS_SM80_BF16_MMA_GEMM
#define NVIDIA_KERNELS_SM80_BF16_MMA_GEMM

#include "kernels/hilbert.cuh"
#include "kernels/load_async.cuh"
#include "kernels/morton_encoding.cuh"
#include "kernels/ring_buffer.cuh"
#include "kernels/utils.cuh"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

using bf16_t = __nv_bfloat16;

namespace nvidia::kernels::sm80 {
template <int BM, int BN, int BK, int WM, int WN, int NumBuffers,
          int NumThreads>
__launch_bounds__(NumThreads, 1) __global__
    void bf16_mma_gemm(const bf16_t *a, const bf16_t *b, const float *c,
                       float *d, int32_t m, int32_t n, int32_t k,
                       const float alpha, const float beta) {
  using T = bf16_t;
  static_assert(WN == 64);
  static_assert(WM == 32);
  extern __shared__ char smem[];
  int lane_id = threadIdx.x % 32;
  int num_matrix = lane_id / 8;
  int warp_id = threadIdx.x / 32;
  // Pad to ensure alignment (multiple of 256 bytes for safety)
  constexpr int32_t a_buffer_size = NumBuffers * BM * BK * sizeof(T);
  constexpr int32_t b_buffer_size = NumBuffers * BK * BN * sizeof(T);
  uint32_t smem_a_addr = static_cast<int32_t>(__cvta_generic_to_shared(smem));
  uint32_t smem_b_addr = smem_a_addr + a_buffer_size;
  uint32_t smem_c_addr = smem_b_addr + b_buffer_size +
                         warp_id * WN * static_cast<int32_t>(sizeof(float));

  int head = 0;
  int tail = 0;
  int k_load_index = 0;

  int32_t m_tiles = (m + BM - 1) / BM;
  int32_t n_tiles = (n + BN - 1) / BN;
  int32_t block_id = blockIdx.x;

  constexpr int MMA_M = 16;
  constexpr int MMA_K = 8;
  constexpr int MMA_N = 8;

  constexpr int num_warps_per_row = BN / WN;
  int warp_row = warp_id / num_warps_per_row;
  int warp_col = warp_id % num_warps_per_row;

  const int smem_a_base_index =
      (warp_row * WM + num_matrix * 8 + (lane_id % 8)) * BK;
  const int smem_b_base_index =
      (lane_id % 8) * BN + warp_col * WN + num_matrix * 8;

  uint32_t block_row;
  uint32_t block_col;
  utils::map_to_hilbert(block_id, n_tiles, block_col, block_row);

#pragma unroll
  for (int i = 0; i < NumBuffers; i++) {
    if (k_load_index < k) {
      async_load::load_swizzled<T, BM, BK, NumThreads>(
          a, smem_a_addr + tail * BM * BK * static_cast<int32_t>(sizeof(T)), k,
          block_row * BM, k_load_index);
      async_load::load_swizzled<T, BK, BN, NumThreads>(
          b, smem_b_addr + tail * BK * BN * static_cast<int32_t>(sizeof(T)), n,
          k_load_index, block_col * BN);

      asm volatile("cp.async.commit_group;\n");
      tail = (tail + 1) % NumBuffers;
      k_load_index += BK;
    }
  }

  while (block_id < m_tiles * n_tiles) {
    constexpr int ChunkLoads = WM;
    float bias_regs[ChunkLoads][2];
    const auto base_output_address =
        (static_cast<int64_t>(block_row) * BM + warp_row * WM) * n +
        block_col * BN + warp_col * WN + 2 * lane_id;
    const auto base_bias_address = c + base_output_address;

#pragma unroll
    for (int i = 0; i < ChunkLoads; i++) {
      asm volatile("ld.global.cs.v2.f32 {%0, %1}, [%2]; \n"
                   : "=f"(bias_regs[i][0]), "=f"(bias_regs[i][1])
                   : "l"(base_bias_address + i * n));
    }

    float c_regs[WM / MMA_M][WN / MMA_N][4];
    head = 0;

    for (int i = 0; i < WM / MMA_M; i++) {
      for (int j = 0; j < WN / MMA_N; j++) {
        for (int l = 0; l < 4; l++) {
          c_regs[i][j][l] = 0;
        }
      }
    }

    for (int kk = 0; kk < k; kk += BK) {
      uint32_t a_regs[WM / MMA_M][2][2];
      uint32_t b_regs[WN / MMA_N][2][1];
      const int smem_base_a_addr =
          smem_a_addr + head * BM * BK * static_cast<int32_t>(sizeof(T));
      const int smem_base_b_addr =
          smem_b_addr + head * BK * BN * static_cast<int32_t>(sizeof(T));
      asm volatile("cp.async.wait_group %0; \n" ::"n"(NumBuffers - 1));
      __syncthreads();

#pragma unroll
      for (int _k = 0; _k < BK / MMA_K; _k++) {
        constexpr int NumBanks = 32;
        constexpr int BankSize = 4;
        constexpr int elements_per_copy = 16 / sizeof(T);
        constexpr int M = utils::log2_floor(elements_per_copy);
        constexpr int B =
            utils::log2_floor((NumBanks * BankSize) / sizeof(T)) - M;
        constexpr int S_A = utils::log2_floor(BK) - M;
        constexpr int S_B = utils::log2_floor(BN) - M;

        // WM = 64; MMA_M = 16 => 2 A matrix loads
        // WN = 128; MMA_N = 8; => 16 / 4 B matrix loads

        int smem_a_index = smem_a_base_index + _k * MMA_K;
        int smem_b_index = smem_b_base_index + _k * MMA_K * BN;

        // Load 0: ---> End of A matrix loads
        int swizzled_address_a = utils::swizzle<B, M, S_A>(smem_a_index) *
                                 static_cast<int32_t>(sizeof(T));
        int swizzled_address_b = utils::swizzle<B, M, S_B>(smem_b_index) *
                                 static_cast<int32_t>(sizeof(T));
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, "
            "%3}, [%4];\n"
            : "=r"(a_regs[0][_k % 2][0]), "=r"(a_regs[0][_k % 2][1]),
              "=r"(a_regs[1][_k % 2][0]), "=r"(a_regs[1][_k % 2][1])
            : "r"(smem_base_a_addr + swizzled_address_a));
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 "
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(b_regs[0][_k % 2][0]), "=r"(b_regs[1][_k % 2][0]),
              "=r"(b_regs[2][_k % 2][0]), "=r"(b_regs[3][_k % 2][0])
            : "r"(smem_base_b_addr + swizzled_address_b));

        // Load 1: ---> End of A matrix loads
        smem_b_index += 4 * MMA_N;
        swizzled_address_b = utils::swizzle<B, M, S_B>(smem_b_index) *
                             static_cast<int32_t>(sizeof(T));
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 "
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(b_regs[4][_k % 2][0]), "=r"(b_regs[5][_k % 2][0]),
              "=r"(b_regs[6][_k % 2][0]), "=r"(b_regs[7][_k % 2][0])
            : "r"(smem_base_b_addr + swizzled_address_b));

#pragma unroll
        for (int _n = 0; _n < WN / MMA_N; _n++) {
#pragma unroll
          for (int _m = 0; _m < WM / MMA_M; _m++) {
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5}, "
                "{%6}, "
                "{%0, %1, %2, %3}; \n"
                : "+f"(c_regs[_m][_n][0]), "+f"(c_regs[_m][_n][1]),
                  "+f"(c_regs[_m][_n][2]), "+f"(c_regs[_m][_n][3])
                : "r"(a_regs[_m][_k % 2][0]), "r"(a_regs[_m][_k % 2][1]),
                  "r"(b_regs[_n][_k % 2][0]));
          }
        }
      }

      head = (head + 1) % NumBuffers;
      if (k_load_index < k) {
        async_load::load_swizzled<T, BM, BK, NumThreads>(
            a, smem_a_addr + tail * BM * BK * static_cast<int32_t>(sizeof(T)),
            k, block_row * BM, k_load_index);
        async_load::load_swizzled<T, BK, BN, NumThreads>(
            b, smem_b_addr + tail * BK * BN * static_cast<int32_t>(sizeof(T)),
            n, k_load_index, block_col * BN);

        asm volatile("cp.async.commit_group;\n");
        tail = (tail + 1) % NumBuffers;
        k_load_index += BK;
      }
    }

    // Due to lack of registers, Do not load everything at once.
    const auto base_d_address = d + base_output_address;

    block_id += gridDim.x;
    // This actually ever so slightly degrades performance. I am yet to
    // understand why
    if (block_id < m_tiles * n_tiles) {
      tail = 0;
      k_load_index = 0;
      utils::map_to_hilbert(block_id, n_tiles, block_col, block_row);

#pragma unroll
      for (int i = 0; i < NumBuffers; i++) {
        if (k_load_index < k) {
          async_load::load_swizzled<T, BM, BK, NumThreads>(
              a, smem_a_addr + tail * BM * BK * static_cast<int32_t>(sizeof(T)),
              k, block_row * BM, k_load_index);
          async_load::load_swizzled<T, BK, BN, NumThreads>(
              b, smem_b_addr + tail * BK * BN * static_cast<int32_t>(sizeof(T)),
              n, k_load_index, block_col * BN);

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
                       "f"(c_regs[i / MMA_M][j / MMA_N][reg_number * 2]),
                       "f"(c_regs[i / MMA_M][j / MMA_N][reg_number * 2 + 1])
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
                     : "l"(base_bias_address + (i + ChunkLoads) * n));
      }
    }
  }
}
}  // namespace nvidia::kernels::sm80

#endif
