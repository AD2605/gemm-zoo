#ifndef NVIDIA_KERNELS_DOUBLE_BUFFERED_GEMM_CUH
#define NVIDIA_KERNELS_DOUBLE_BUFFERED_GEMM_CUH

#include "kernels/ring_buffer.cuh"
#include "kernels/utils.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace nvidia::kernels {
template <typename TIn, typename TOut, int M, int N, int K, int TM, int TN,
          int TK, int BlockDim, int NumBuffers>
__launch_bounds__(BlockDim) __global__
    void double_buffered_gemm(const TIn* a, const TIn* b, const TOut* c,
                              TOut* d, std::size_t m, std::size_t n,
                              std::size_t k, TOut alpha, TOut beta) {
  static_assert(std::is_same_v<TIn, float>);
  static_assert(std::is_same_v<TOut, float>);

  static_assert(K % 2 == 0);
  //  static_assert(TM % 16 == 0);
  static_assert(TN % 4 == 0);
  static_assert(N % 128 == 0);
  static_assert(BlockDim == 256);
  static_assert(NumBuffers == 2);

  extern __shared__ char smem[];
  constexpr int32_t sizeof_TIn = static_cast<int32_t>(sizeof(TIn));
  int32_t smem_a_addr = static_cast<int32_t>(__cvta_generic_to_shared(smem));
  int32_t smem_b_addr = smem_a_addr + M * K * sizeof_TIn * NumBuffers;

  ring_buffer::smem_ring_buffer<NumBuffers, M * K * sizeof_TIn> smem_a_buffer(
      smem_a_addr);
  ring_buffer::smem_ring_buffer<NumBuffers, K * N * sizeof_TIn> smem_b_buffer(
      smem_b_addr);

  int block_id = blockIdx.x;
  int warp_id = threadIdx.x / 32;
  constexpr int num_warps = BlockDim / 32;
  constexpr int warp_size = 32;
  const int thread_id = threadIdx.x % 32;

  TOut RmemD[TM * TN];
  TIn RmemA[TM * TK];
  TIn RmemB[TK * TN];
  TOut RmemC[4];  // variables in which c values will be loaded;

  const int M_tiles = utils::ceil_div(m, M);
  const int N_tiles = utils::ceil_div(n, N);
  const int total_tiles = M_tiles * N_tiles;

  for (; block_id < total_tiles; block_id += gridDim.x) {
#pragma unroll
    for (int i = 0; i < TM * TN; i++) {
      RmemD[i] = 0;
    }

    auto tile_row_start = block_id / N_tiles;
    auto tile_col_start = block_id % N_tiles;

    // just hardcoded for now
    for (int kk = 0; kk < k; kk += K) {
      auto smem_a_k_addr = smem_a_buffer.get_current_buffer();
      auto smem_b_k_addr = smem_b_buffer.get_current_buffer();

      // Load SmemA tile
      for (int m_warp = warp_id; m_warp < M; m_warp += num_warps) {
#pragma unroll
        for (int k_warp = 0; k_warp < K; k_warp += warp_size) {
          auto col = k_warp + thread_id;
          auto gmem_offset = (tile_row_start * M + m_warp) * k + (kk + col);
          auto smem_offset = (m_warp * K + col) * sizeof_TIn;
          asm volatile(
              "{\n\t"
              ".reg .f32 v0; \n\t"
              "ld.global.ca.L2::256B.f32 v0, [%0]; \n\t"
              "st.shared.f32 [%1], v0; \n"
              "}"
              :
              : "l"(a + gmem_offset), "r"(smem_a_k_addr + smem_offset)
              : "memory");
        }
      }

      // Load SmemB Tile;
      for (int k_warp = warp_id; k_warp < K; k_warp += num_warps) {
#pragma unroll
        for (int n_warp = 0; n_warp < N; n_warp += warp_size * 4) {
          auto col = n_warp + thread_id * 4;
          auto gmem_offset = (kk + k_warp) * n + (tile_col_start * N + col);
          auto smem_offset = (k_warp * N + col) * sizeof_TIn;
          asm volatile(
              "{\n\t"
              ".reg .f32 v0, v1, v2, v3; \n\t"
              "ld.global.ca.L2::256B.v4.f32 {v0, v1, v2, v3}, [%0]; \n\t"
              "st.shared.v4.f32 [%1], {v0, v1, v2, v3}; \n"
              "}"
              :
              : "l"(b + gmem_offset), "r"(smem_b_k_addr + smem_offset)
              : "memory");
        }
      }
      __syncthreads();

      for (int inner = 0; inner < K; inner += TK) {
#pragma unroll
        for (int mm = 0; mm < TM; mm++) {
          auto smem_row_offset = (warp_id * TM + mm) * K + inner;
#pragma unroll
          for (int k_thread = 0; k_thread < TK; k_thread += 2) {
            asm volatile(
                "{\n\t"
                "ld.shared.v2.f32 {%0, %1}, [%2]; \n"
                "}"
                : "=f"(RmemA[mm * TK + k_thread + 0]),
                  "=f"(RmemA[mm * TK + k_thread + 1])
                : "r"(smem_a_addr + (smem_row_offset + k_thread) * sizeof_TIn));
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
                : "r"(smem_b_addr +
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
}  // namespace nvidia::kernels

#endif
