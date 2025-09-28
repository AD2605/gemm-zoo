

#ifndef NVIDIA_KERNELS_SM80_TF32_MMA_GEMM_CUH
#define NVIDIA_KERNELS_SM80_TF32_MMA_GEMM_CUH

#include "kernels/ring_buffer.cuh"
#include "kernels/utils.cuh"

#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
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
          "cp.async.ca.shared.global.L2::256B [%0], [%1], 4; \n\t"
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
          "cp.async.ca.shared.global.L2::256B [%0], [%1], 16; \n\t"
          "}"
          :
          : "r"(smem_b_addr + smem_offset), "l"(gmem_ptr + gmem_offset)
          : "memory");
    }
  }
}

template <typename T>
__device__ __forceinline__ void load_matrix_a(const T* smem_a, const T* smem_b,
                                              T* a_regs, T* b_regs,
                                              int32_t smem_a_ld,
                                              int32_t smem_a_col_offset) {
  // hardcoded for m16n8k8 tf32 mma
  T a0, a1, a2, a3;

  auto t_idx = threadIdx.x % 32;
  int group_id = t_idx >> 2;
  int threadID_in_group = t_idx % 4;

  a0 = smem_a[group_id * smem_a_ld + threadID_in_group + smem_a_col_offset];
  a2 = smem_a[group_id * smem_a_ld + threadID_in_group + 4 + smem_a_col_offset];
  a1 = smem_a[(group_id + 8) * smem_a_ld + threadID_in_group +
              smem_a_col_offset];
  a3 = smem_a[(group_id + 8) * smem_a_ld + threadID_in_group + 4 +
              smem_a_col_offset];
}

template <typename T>
__device__ __forceinline__ void load_matrix_b(const T* smem_b, T* b_regs,
                                              int32_t smem_b_ld,
                                              int32_t smem_b_col_offset) {
  T b0, b1;

  auto t_idx = threadIdx.x % 32;
  int group_id = t_idx >> 2;
  int threadID_in_group = t_idx % 4;

  b0 = smem_b[threadID_in_group * smem_b_ld + group_id + smem_b_col_offset];
  b1 = smem_b[(threadID_in_group + 4) * smem_b_ld + group_id +
              smem_b_col_offset];
}

}  // namespace detail

template <typename TIn, typename TOut, int M, int N, int K, int BlockDim,
          int NumBuffers>
__launch_bounds__(BlockDim) __global__
    void async_buffered_gemm(const TIn* a, const TIn* b, const TOut* c, TOut* d,
                             std::size_t m, std::size_t n, std::size_t k,
                             TOut alpha, TOut beta) {
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
  int warp_id = threadIdx.x / 32;
  constexpr int num_warps = BlockDim / 32;
  const int thread_id = threadIdx.x % 32;

  const int M_tiles = utils::ceil_div(m, M);
  const int N_tiles = utils::ceil_div(n, N);
  const int total_tiles = M_tiles * N_tiles;

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

      int32_t smem_a_k_addr = smem_a_addr + head * M * K * sizeof_TIn;
      int32_t smem_b_k_addr = smem_b_addr + head * K * N * sizeof_TIn;

      // m16n8k8
      TIn A_regs[WM / MMA_M][WK / MMA_K][TM];  // 4 x 2 x 4
      TIn B_regs[WK / MMA_K][WN / MMA_N][TN];  // 2 x 4 x 2

      constexpr int num_warps_per_row = 4;
      const int warp_id = threadIdx.x / 32;
      auto warp_tile_row_id = warp_id / 4;
      auto warp_tile_col_id = warp_id % 4;
      auto smem_a_col_offset = warp_tile_col_id * 4 * 32;

#pragma unroll
      for (int inner = 0; inner < K; inner += WK) {
        for (int i = 0; i < WM; i += MMA_M) {
          for (int j = 0; j < WK; j += MMA_K) {
          }
        }
      }
      __syncthreads();

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

#pragma unroll
    for (int m_thread = 0; m_thread < TM; m_thread++) {
      auto row = warp_id * TM + m_thread;
#pragma unroll
      for (int n_thread = 0; n_thread < TN; n_thread += 4) {
        auto col = thread_id * TN + n_thread;
        auto output_offset =
            (tile_row_start * M + row) * n + tile_col_start * N + col;
        asm volatile(
            "{\n\t"
            "ld.global.v4.f32 {%0, %1, %2, %3}, [%4]; \n"
            "}"
            : "=f"(RmemC[0]), "=f"(RmemC[1]), "=f"(RmemC[2]), "=f"(RmemC[3])
            : "l"(c + output_offset));

#pragma unroll
        for (int i = 0; i < 4; i++) {
          RmemD[m_thread * TN + n_thread + i] =
              alpha * RmemD[m_thread * TN + n_thread + i] + beta * RmemC[i];
        }

        asm volatile(
            "{\n\t"
            "st.global.v4.f32 [%0], {%1, %2, %3, %4}; \n"
            "}" ::"l"(d + output_offset),
            "f"(RmemD[m_thread * TN + n_thread + 0]),
            "f"(RmemD[m_thread * TN + n_thread + 1]),
            "f"(RmemD[m_thread * TN + n_thread + 2]),
            "f"(RmemD[m_thread * TN + n_thread + 3])
            : "memory");
      }
    }
  }
}
}  // namespace nvidia::kernels::sm80

#endif
