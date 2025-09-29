#ifndef NVIDIA_KERNELS_SM80_TF32_MMA_GEMM_CUH
#define NVIDIA_KERNELS_SM80_TF32_MMA_GEMM_CUH

#include "kernels/ring_buffer.cuh"
#include "kernels/utils.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace nvidia::kernels::sm80 {

namespace detail {
template <int M, int K, typename TIn>
__device__ __forceinline__ void async_populate_smemA_buffer(
    const int32_t smem_a_addr, const TIn* gmem_ptr, const int num_warps,
    const int32_t warp_id, const int32_t thread_id, const int tile_row_start,
    const int kk, int k) {
  for (int m_warp = warp_id; m_warp < M; m_warp += num_warps) {
#pragma unroll
    for (int k_warp = 0; k_warp < K; k_warp += 32) {
      auto col = k_warp + thread_id;
      auto gmem_offset = (tile_row_start * M + m_warp) * k + (kk + col);
      auto smem_offset = (m_warp * K + col) * static_cast<int32_t>(sizeof(TIn));
      asm volatile(
          "{\n\t"
          "cp.async.ca.shared.global.L2::128B [%0], [%1], 4; \n\t"
          "}"
          :
          : "r"(smem_a_addr + smem_offset), "l"(gmem_ptr + gmem_offset)
          : "memory");
    }
  }
}

template <int K, int N, typename TIn>
__device__ __forceinline__ void async_populate_smemB_buffer(
    const int32_t smem_b_addr, const TIn* gmem_ptr, const int num_warps,
    const int32_t warp_id, const int32_t thread_id, const int tile_col_start,
    const int kk, int n) {
  for (int k_warp = warp_id; k_warp < K; k_warp += num_warps) {
#pragma unroll
    for (int n_warp = 0; n_warp < N; n_warp += 32 * 4) {
      auto col = n_warp + thread_id * 4;
      auto gmem_offset = (kk + k_warp) * n + (tile_col_start * N + col);
      auto smem_offset = (k_warp * N + col) * static_cast<int32_t>(sizeof(TIn));
      asm volatile(
          "{\n\t"
          "cp.async.ca.shared.global.L2::128B [%0], [%1], 16; \n\t"
          "}"
          :
          : "r"(smem_b_addr + smem_offset), "l"(gmem_ptr + gmem_offset)
          : "memory");
    }
  }
}

}  // namespace detail

template <typename TIn, typename TOut, int M, int N, int K, int BlockDim,
          int NumBuffers>
__launch_bounds__(BlockDim) __global__
    void tf32_mma_gemm(const TIn* a, const TIn* b, const TOut* c, TOut* d,
                       std::size_t m, std::size_t n, std::size_t k, TOut alpha,
                       TOut beta) {
  static_assert(std::is_same_v<TIn, float>);
  static_assert(std::is_same_v<TOut, float>);

  static_assert(K % 2 == 0);
  static_assert(N % 128 == 0);
  static_assert(BlockDim == 256);

  extern __shared__ char smem[];
  constexpr int32_t sizeof_TIn = static_cast<int32_t>(sizeof(TIn));
  int32_t smem_a_addr = static_cast<int32_t>(__cvta_generic_to_shared(smem));
  int32_t smem_b_addr = smem_a_addr + M * K * sizeof_TIn * NumBuffers;

  int block_id = blockIdx.x;
  constexpr int num_warps = BlockDim / 32;
  const int thread_id = threadIdx.x % 32;

  const int M_tiles = utils::ceil_div(m, M);
  const int N_tiles = utils::ceil_div(n, N);
  const int total_tiles = M_tiles * N_tiles;

  auto t_idx = threadIdx.x % 32;

  int group_id = t_idx >> 2;
  int threadID_in_group = t_idx % 4;

  for (; block_id < total_tiles; block_id += gridDim.x) {
    constexpr int WM = 64;
    constexpr int WN = 32;
    constexpr int WK = 16;

    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 8;

    constexpr int TM = 4;
    constexpr int TN = 2;
    constexpr int TC = 4;

    TOut C_regs[WM / MMA_M][WN / MMA_N][TC];  // 4 x 4 x 4
    for (int _i = 0; _i < WM / MMA_M; _i++) {
      for (int _j = 0; _j < WN / MMA_N; _j++) {
        for (int _k = 0; _k < TC; _k++) {
          C_regs[_i][_j][_k] = 0;
        }
      }
    }

    auto tile_row_start = block_id / N_tiles;
    auto tile_col_start = block_id % N_tiles;

    const int warp_id = threadIdx.x / 32;
    auto warp_tile_row_id = warp_id / 4;
    auto warp_tile_col_id = warp_id % 4;

    int head = 0;
    int tail = 0;

    // start the transfer of all the buffers;
    int k_load_index = 0;
#pragma unroll(NumBuffers)
    for (int i = 0; i < NumBuffers; i++) {
      if (k_load_index < k) {
        detail::async_populate_smemA_buffer<M, K, TIn>(
            smem_a_addr + tail * M * K * sizeof_TIn, a, num_warps, warp_id,
            thread_id, tile_row_start, k_load_index, k);
        detail::async_populate_smemB_buffer<K, N, TIn>(
            smem_b_addr + tail * K * N * sizeof_TIn, b, num_warps, warp_id,
            thread_id, tile_col_start, k_load_index, n);
        asm volatile("cp.async.commit_group;\n");
        k_load_index += K;
        tail = (tail + 1) % NumBuffers;
      }
    }

    for (int kk = 0; kk < k; kk += K) {
      asm volatile("cp.async.wait_group 1; \n");
      __syncthreads();

      TIn* smem_a_ptr = reinterpret_cast<TIn*>(smem) + head * M * K;
      TIn* smem_b_ptr =
          reinterpret_cast<TIn*>(smem) + NumBuffers * M * K + head * K * N;
      // m16n8k8
      TIn A_regs[WK / MMA_K][WM / MMA_M][TM];  // 2 x 4 x 4
      TIn B_regs[WK / MMA_K][WN / MMA_N][TN];  // 2 x 4 x 2

#pragma unroll
      for (int inner = 0; inner < K; inner += WK) {
#pragma unroll
        for (int i = 0; i < WM / MMA_M; i++) {
#pragma unroll
          for (int j = 0; j < WK / MMA_K; j++) {
            A_regs[j][i][0] =
                smem_a_ptr[(warp_tile_row_id * WM + i * MMA_M + group_id) * K +
                           inner + j * MMA_K + threadID_in_group];
            A_regs[j][i][2] =
                smem_a_ptr[(warp_tile_row_id * WM + i * MMA_M + group_id) * K +
                           inner + j * MMA_K + threadID_in_group + 4];
            A_regs[j][i][1] =
                smem_a_ptr[(warp_tile_row_id * WM + i * MMA_M + group_id + 8) *
                               K +
                           inner + j * MMA_K + threadID_in_group];
            A_regs[j][i][3] =
                smem_a_ptr[(warp_tile_row_id * WM + i * MMA_M + group_id + 8) *
                               K +
                           inner + j * MMA_K + threadID_in_group + 4];
          }
        }

#pragma unroll
        for (int i = 0; i < WK / MMA_K; i++) {
#pragma unroll
          for (int j = 0; j < WN / MMA_N; j++) {
            B_regs[i][j][0] =
                smem_b_ptr[(inner + i * MMA_K + threadID_in_group) * N +
                           warp_tile_col_id * WN + j * MMA_N + group_id];
            B_regs[i][j][1] =
                smem_b_ptr[(inner + i * MMA_K + threadID_in_group + 4) * N +
                           warp_tile_col_id * WN + j * MMA_N + group_id];
          }
        }

// 2 x 4 x 4 = 32 MMAs
#pragma unroll
        for (int _k = 0; _k < WK / MMA_K; _k++) {
#pragma unroll
          for (int _m = 0; _m < WM / MMA_M; _m++) {
#pragma unroll
            for (int _n = 0; _n < WN / MMA_N; _n++) {
              asm volatile(
                  "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                  "{%0, %1, %2, %3}, "
                  "{%4, %5, %6, %7}, "
                  "{%8, %9}, "
                  "{%10, %11, %12, %13}; \n"
                  : "=f"(C_regs[_m][_n][0]), "=f"(C_regs[_m][_n][1]),
                    "=f"(C_regs[_m][_n][2]), "=f"(C_regs[_m][_n][3])
                  : "r"(*reinterpret_cast<int32_t*>(&A_regs[_k][_m][0])),
                    "r"(*reinterpret_cast<int32_t*>(&A_regs[_k][_m][1])),
                    "r"(*reinterpret_cast<int32_t*>(&A_regs[_k][_m][2])),
                    "r"(*reinterpret_cast<int32_t*>(&A_regs[_k][_m][3])),
                    "r"(*reinterpret_cast<int32_t*>(&B_regs[_k][_n][0])),
                    "r"(*reinterpret_cast<int32_t*>(&B_regs[_k][_n][1])),
                    "f"(C_regs[_m][_n][0]), "f"(C_regs[_m][_n][1]),
                    "f"(C_regs[_m][_n][2]), "f"(C_regs[_m][_n][3]));
            }
          }
        }
      }
      //__syncthreads();

      head = (head + 1) % NumBuffers;

      // populate the (k + 3)th buffer;
      if (k_load_index < k) {
        detail::async_populate_smemA_buffer<M, K, TIn>(
            smem_a_addr + tail * M * K * sizeof_TIn, a, num_warps, warp_id,
            thread_id, tile_row_start, k_load_index, k);
        detail::async_populate_smemB_buffer<K, N, TIn>(
            smem_b_addr + tail * K * N * sizeof_TIn, b, num_warps, warp_id,
            thread_id, tile_col_start, k_load_index, n);
        asm volatile("cp.async.commit_group;\n");
        k_load_index += K;
        tail = (tail + 1) % NumBuffers;
      }
    }

    // Pipeline loads from global Memory;
    // Output Tile Size M x N;
    constexpr int NumWarps = 8;
    TOut D_regs[M / NumWarps][N / 32];

#pragma unroll
    for (int i = 0; i < M / NumWarps; i++) {
#pragma unroll
      for (int j = 0; j < N; j += 128) {
        int loc_row = i * NumWarps + warp_id;
        int loc_col = t_idx * 4 + j;
        int global_row = tile_row_start * M + loc_row;
        int global_col = tile_col_start * N + loc_col;
        asm volatile(
            "ld.global.v4.f32 {%0, %1, %2, %3}, [%4]; \n"
            : "=f"(D_regs[i][(j / 32) + 0]), "=f"(D_regs[i][(j / 32) + 1]),
              "=f"(D_regs[i][(j / 32) + 2]), "=f"(D_regs[i][(j / 32) + 3])
            : "l"(c + global_row * n + global_col));
      }
    }

    // Now start storing everything back to shared memory;
    TOut* output_ptr = reinterpret_cast<TOut*>(smem);  // Reuse the same Smem
    for (int i = 0; i < WM / MMA_M; i++) {
      for (int j = 0; j < WN / MMA_N; j++) {
        output_ptr[(warp_tile_row_id * WM + i * MMA_M + group_id) * N +
                   warp_tile_col_id * WN + j * MMA_N +
                   ((threadID_in_group * 2) + (0 & 0x1))] = C_regs[i][j][0];
        output_ptr[(warp_tile_row_id * WM + i * MMA_M + group_id) * N +
                   warp_tile_col_id * WN + j * MMA_N +
                   ((threadID_in_group * 2) + (1 & 0x1))] = C_regs[i][j][1];
        output_ptr[(warp_tile_row_id * WM + i * MMA_M + group_id + 8) * N +
                   warp_tile_col_id * WN + j * MMA_N +
                   ((threadID_in_group * 2) + (2 & 0x1))] = C_regs[i][j][2];
        output_ptr[(warp_tile_row_id * WM + i * MMA_M + group_id + 8) * N +
                   warp_tile_col_id * WN + j * MMA_N +
                   ((threadID_in_group * 2) + (3 & 0x1))] = C_regs[i][j][3];
      }
    }
    __syncthreads();
    TOut D_regs_temp[M / NumWarps][N / 32];
    int32_t output_ptr_addr =
        static_cast<int32_t>(__cvta_generic_to_shared(output_ptr));

#pragma unroll
    for (int i = 0; i < M / NumWarps; i++) {
#pragma unroll
      for (int j = 0; j < N; j += 128) {
        asm volatile(
            "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4]; \n"
            : "=f"(D_regs_temp[i][(j / 32) + 0]),
              "=f"(D_regs_temp[i][(j / 32) + 1]),
              "=f"(D_regs_temp[i][(j / 32) + 2]),
              "=f"(D_regs_temp[i][(j / 32) + 3])
            : "r"(output_ptr_addr + (i * NumWarps + warp_id) * N * sizeof_TIn +
                  (t_idx * 4 + j) * sizeof_TIn));
      }
    }

#pragma unroll
    for (int i = 0; i < M / NumWarps; i++) {
#pragma unroll
      for (int jj = 0; jj < N / 32; jj++) {
        D_regs_temp[i][jj] = alpha * D_regs_temp[i][jj] + beta * D_regs[i][jj];
      }
    }

#pragma unroll
    for (int i = 0; i < M / NumWarps; i++) {
#pragma unroll
      for (int j = 0; j < N; j += 128) {
        int loc_row = i * NumWarps + warp_id;
        int loc_col = t_idx * 4 + j;
        int global_row = tile_row_start * M + loc_row;
        int global_col = tile_col_start * N + loc_col;
        asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4}; \n" ::"l"(
                         d + global_row * n + global_col),
                     "f"(D_regs_temp[i][(j / 32) + 0]),
                     "f"(D_regs_temp[i][(j / 32) + 1]),
                     "f"(D_regs_temp[i][(j / 32) + 2]),
                     "f"(D_regs_temp[i][(j / 32) + 3])
                     : "memory");
      }
    }
  }
}
}  // namespace nvidia::kernels::sm80

#endif
