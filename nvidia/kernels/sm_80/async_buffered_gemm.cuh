#ifndef NVIDIA_KERNELS_SM80_ASYNC_BUFFERED_GEMM_CUH
#define NVIDIA_KERNELS_SM80_ASYNC_BUFFERED_GEMM_CUH

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
}  // namespace detail

template <typename TIn, typename TOut, int M, int N, int K, int TM, int TN,
          int TK, int BlockDim, int NumBuffers>
__launch_bounds__(BlockDim) __global__
    void async_buffered_gemm(const TIn* a, const TIn* b, const TOut* c, TOut* d,
                             std::size_t m, std::size_t n, std::size_t k,
                             TOut alpha, TOut beta) {
  static_assert(std::is_same_v<TIn, float>);
  static_assert(std::is_same_v<TOut, float>);

  static_assert(K % 2 == 0);
  //  static_assert(TM % 16 == 0);
  static_assert(TN % 4 == 0);
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

  TOut RmemD[TM * TN];
  TIn RmemA[TM * TK];
  TIn RmemB[TK * TN];

  const int M_tiles = utils::ceil_div(m, M);
  const int N_tiles = utils::ceil_div(n, N);
  const int total_tiles = M_tiles * N_tiles;

  auto tile_row_start = block_id / N_tiles;
  auto tile_col_start = block_id % N_tiles;

  int tile_row_start_temp;
  int tile_col_start_temp;

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

  while (block_id < total_tiles) {
#pragma unroll
    for (int i = 0; i < TM * TN; i++) {
      RmemD[i] = 0;
    }

    for (int kk = 0; kk < k; kk += K) {
      asm volatile("cp.async.wait_group 1; \n");
      __syncthreads();

      int32_t smem_a_k_addr = smem_a_addr + head * M * K * sizeof_TIn;
      int32_t smem_b_k_addr = smem_b_addr + head * K * N * sizeof_TIn;

#pragma unroll
      for (int inner = 0; inner < K; inner += TK) {
#pragma unroll
        for (int mm = 0; mm < TM; mm++) {
          auto smem_row_offset = (warp_id * TM + mm) * K + inner;
#pragma unroll
          for (int k_thread = 0; k_thread < TK; k_thread += 4) {
            asm volatile(
                "{\n\t"
                "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4]; \n"
                "}"
                : "=f"(RmemA[mm * TK + k_thread + 0]),
                  "=f"(RmemA[mm * TK + k_thread + 1]),
                  "=f"(RmemA[mm * TK + k_thread + 2]),
                  "=f"(RmemA[mm * TK + k_thread + 3])
                : "r"(smem_a_k_addr +
                      (smem_row_offset + k_thread) * sizeof_TIn));
          }
        }

#pragma unroll
        for (int k_thread = 0; k_thread < TK; k_thread++) {
          int32_t smem_b_offset = (inner + k_thread) * N;
#pragma unroll
          for (int n_thread = 0; n_thread < TN; n_thread += 4) {
            asm volatile(
                "{\n\t"
                "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4]; \n"
                "}"
                : "=f"(RmemB[k_thread * TN + n_thread + 0]),
                  "=f"(RmemB[k_thread * TN + n_thread + 1]),
                  "=f"(RmemB[k_thread * TN + n_thread + 2]),
                  "=f"(RmemB[k_thread * TN + n_thread + 3])
                : "r"(smem_b_k_addr +
                      (smem_b_offset + thread_id * TN + n_thread) *
                          sizeof_TIn));
          }
        }

#pragma unroll
        for (int k_thread = 0; k_thread < TK; k_thread++) {
#pragma unroll
          for (int m_thread = 0; m_thread < TM; m_thread++) {
#pragma unroll
            for (int n_thread = 0; n_thread < TN; n_thread++) {
              RmemD[m_thread * TN + n_thread] +=
                  RmemA[m_thread * TK + k_thread] *
                  RmemB[k_thread * TN + n_thread];
            }
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

    TOut RmemC[TM][TN];  // variables in which c values will be loaded;

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
            : "=f"(RmemC[m_thread][n_thread + 0]),
              "=f"(RmemC[m_thread][n_thread + 1]),
              "=f"(RmemC[m_thread][n_thread + 2]),
              "=f"(RmemC[m_thread][n_thread + 3])
            : "l"(c + output_offset));
      }
    }

    block_id += gridDim.x;
    if (block_id < total_tiles) {
      tile_row_start_temp = block_id / N_tiles;
      tile_col_start_temp = block_id % N_tiles;
      head = 0;
      tail = 0;

      // start the transfer of all the buffers;
      k_load_index = 0;
#pragma unroll(NumBuffers)
      for (int i = 0; i < NumBuffers; i++) {
        if (k_load_index < k) {
          detail::async_populate_smemA_buffer<M, K, TIn>(
              smem_a_addr + tail * M * K * sizeof_TIn, a, num_warps, warp_id,
              thread_id, tile_row_start_temp, k_load_index, k);
          detail::async_populate_smemB_buffer<K, N, TIn>(
              smem_b_addr + tail * K * N * sizeof_TIn, b, num_warps, warp_id,
              thread_id, tile_col_start_temp, k_load_index, n);
          asm volatile("cp.async.commit_group;\n");
          k_load_index += K;
          tail = (tail + 1) % NumBuffers;
        }
      }
    }

#pragma unroll
    for (int m_thread = 0; m_thread < TM; m_thread++) {
      auto row = warp_id * TM + m_thread;
#pragma unroll
      for (int n_thread = 0; n_thread < TN; n_thread++) {
        RmemD[m_thread * TN + n_thread] =
            alpha * RmemD[m_thread * TN + n_thread] +
            beta * RmemC[m_thread][n_thread];
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
            "st.global.v4.f32 [%0], {%1, %2, %3, %4}; \n"
            "}" ::"l"(d + output_offset),
            "f"(RmemD[m_thread * TN + n_thread + 0]),
            "f"(RmemD[m_thread * TN + n_thread + 1]),
            "f"(RmemD[m_thread * TN + n_thread + 2]),
            "f"(RmemD[m_thread * TN + n_thread + 3])
            : "memory");
      }
    }

    tile_row_start = tile_row_start_temp;
    tile_col_start = tile_col_start_temp;
  }
}
}  // namespace nvidia::kernels::sm80

#endif
