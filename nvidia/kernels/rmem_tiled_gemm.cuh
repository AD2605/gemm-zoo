#ifndef NVIDIA_KERNELS_RMEM_TILED_GEMM_CUH
#define NVIDIA_KERNELS_RMEM_TILED_GEMM_CUH

#include "kernels/utils.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace nvidia::kernels {
template <typename TIn, typename TOut, int M, int N, int K, int TM, int TN,
          int TK>
__global__ void rmem_tiled_gemm(const TIn* a, const TIn* b, const TOut* c,
                                TOut* d, std::size_t m, std::size_t n,
                                std::size_t k, TOut alpha, TOut beta) {
  static_assert(std::is_same_v<TIn, float>);
  static_assert(std::is_same_v<TOut, float>);

  extern __shared__ char smem[];
  int32_t sizeof_TIn = static_cast<int32_t>(sizeof(TIn));
  int32_t smem_a_addr = static_cast<int32_t>(__cvta_generic_to_shared(smem));
  int32_t smem_b_addr = smem_a_addr + M * K * sizeof_TIn;

  const int32_t row_num = threadIdx.x / 32;
  const int32_t col_num = threadIdx.x % 32;

  constexpr int SubTile_H = 32;
  constexpr int SubTile_W = 32;

  static_assert(M % SubTile_H == 0);
  static_assert(N % SubTile_W == 0);

  int block_id = blockIdx.x;

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

    for (int K_tile = 0; K_tile < k; K_tile += K) {
      // first part stays exactly the same as smem_tiled_gemm.
      // load a_tile;
#pragma unroll
      for (int i = 0; i < M; i += SubTile_H) {
#pragma unroll
        for (int j = 0; j < K; j += SubTile_W * 4) {
          auto offset =
              (tile_row_start * M + row_num + i) * k + j + col_num * 4 + K_tile;
          asm volatile(
              "{\n\t"
              ".reg .f32 v0, v1, v2, v3; \n\t"
              "ld.global.ca.L2::256B.v4.f32 {v0, v1, v2, v3}, [%0]; \n\t"
              "st.shared.v4.f32 [%1], {v0, v1, v2, v3}; \n"
              "}"
              :
              : "l"(a + offset),
                "r"(smem_a_addr +
                    ((row_num + i) * K + j + col_num * 4) * sizeof_TIn)
              : "memory");
        }
      }

      // load b_tile;
#pragma unroll
      for (int i = 0; i < K; i += SubTile_H) {
#pragma unroll
        for (int j = 0; j < N; j += SubTile_W * 4) {
          auto offset =
              (K_tile + row_num + i) * n + j + col_num * 4 + tile_col_start * N;
          asm volatile(
              "{\n\t"
              ".reg .f32 v0, v1, v2, v3; \n\t"
              "ld.global.ca.L2::256B.v4.f32 {v0, v1, v2, v3}, [%0]; \n\t"
              "st.shared.v4.f32 [%1], {v0, v1, v2, v3}; \n"
              "}"
              :
              : "l"(b + offset),
                "r"(smem_b_addr +
                    ((row_num + i) * N + j + col_num * 4) * sizeof_TIn)
              : "memory");
        }
      }

      __syncthreads();

      // now each warp is responsible for 2x128 output(this is the warp tile, M
      // x N Ordering)
      //  and each thread in a warp in responsible for 2x4 output (M x N
      //  ordering); and thus each warp is responsible for 2 rows; thus we can
      //  do output row = 2 x row_num; This kernel is implicity warp tiled. load
      //  RmemA and RmemB;

#pragma unroll
      for (int kk = 0; kk < K; kk += TK) {
// Load RMEMA register Tile, All threads in a warp will access the same
// row position, but this should result in a warp level broadcast from shared
// memory (hopefully)
#pragma unroll
        for (int mm = 0; mm < TM; mm++) {
          int32_t smem_a_offset = (row_num * TM + mm) * K + kk;
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
                : "r"(smem_a_addr + (smem_a_offset + k_thread) * sizeof_TIn));
          }
        }

// load B tile;
#pragma unroll
        for (int k_thread = 0; k_thread < TK; k_thread++) {
          int32_t smem_b_offset = (kk + k_thread) * N;
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
                      (smem_b_offset + col_num * TN + n_thread) * sizeof_TIn));
          }
        }

        // inner MMAs via outer product;
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
    }
#pragma unroll
    for (int m_thread = 0; m_thread < TM; m_thread++) {
#pragma unroll
      for (int n_thread = 0; n_thread < TN; n_thread += 4) {
        auto output_offset =
            (tile_row_start * M + row_num * TM + m_thread) * n +
            tile_col_start * N + col_num * TN + n_thread;
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
