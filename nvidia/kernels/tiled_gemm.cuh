#ifndef NVIDIA_KERNELS_TILED_GEMM_CUH
#define NVIDIA_KERNELS_TILED_GEMM_CUH

#include <cstddef>
#include <cuda_runtime.h>

namespace nvidia::kernels {
__device__ __host__ __forceinline__ int ceil_div(int dividend, int divisor) {
  return (dividend + divisor - 1) / divisor;
}

template <typename TIn, typename TOut, int M, int N, int K>
__global__ void smem_tiled_gemm(const TIn* a, const TIn* b, const TOut* c,
                                TOut* d, std::size_t m, std::size_t n,
                                std::size_t k, TOut alpha, TOut beta) {
  extern __shared__ char smem[];

  TIn* smem_a = reinterpret_cast<TIn*>(smem);
  TIn* smem_b = smem_a + M * K;

  constexpr int SubTile_H = 32;
  constexpr int SubTile_W = 32;

  static_assert(M % SubTile_H == 0);
  static_assert(N % SubTile_W == 0);

  constexpr int M_regs = M / 32;
  constexpr int N_regs = N / 32;
  TIn acc[M_regs * N_regs] = {0.0f};

  const int M_tiles = ceil_div(m, M);
  const int N_tiles = ceil_div(n, N);
  const int total_tiles = M_tiles * N_tiles;

  auto block_id = blockIdx.x;
  const int row_num = threadIdx.x / SubTile_H;
  const int col_num = threadIdx.x % 32;

  int32_t smem_a_addr = static_cast<int32_t>(__cvta_generic_to_shared(smem_a));
  int32_t smem_b_addr = smem_a_addr + M * K * sizeof(TIn);

  for (; block_id < total_tiles; block_id += gridDim.x) {
    for (int i = 0; i < M_regs * N_regs; i++) {
      acc[i] = 0;
    }
    auto tile_row_start = block_id / N_tiles;
    auto tile_col_start = block_id % N_tiles;

    for (int K_tile = 0; K_tile < k; K_tile += K) {
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
              "ld.global.v4.f32 {v0, v1, v2, v3}, [%0]; \n\t"
              "st.shared.v4.f32 [%1], {v0, v1, v2, v3}; \n"
              "}"
              :
              : "l"(a + offset),
                "r"(smem_a_addr +
                    static_cast<int32_t>(((row_num + i) * K + j + col_num * 4) *
                                         sizeof(TIn)))
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
              "ld.global.v4.f32 {v0, v1, v2, v3}, [%0]; \n\t"
              "st.shared.v4.f32 [%1], {v0, v1, v2, v3}; \n"
              "}"
              :
              : "l"(b + offset),
                "r"(smem_b_addr +
                    static_cast<int32_t>(((row_num + i) * N + j + col_num * 4) *
                                         sizeof(TIn)))
              : "memory");
        }
      }

      __syncthreads();

#pragma unroll
      for (int ii = 0; ii < M; ii += SubTile_H) {
        auto smem_a_row_offset = (ii + row_num) * K;
#pragma unroll
        for (int kk = 0; kk < K; kk++) {
          for (int jj = 0; jj < N; jj += SubTile_W) {
            const int acc_reg_offset =
                (ii / SubTile_H) * N_regs + (jj / SubTile_W);
            acc[acc_reg_offset] += smem_a[smem_a_row_offset + kk] *
                                   smem_b[kk * N + jj + threadIdx.x % 32];
          }
        }
      }
      __syncthreads();
    }
#pragma unroll
    for (int ii = 0; ii < M; ii += SubTile_H) {
#pragma unroll
      for (int jj = 0; jj < N; jj += SubTile_W) {
        const int acc_reg_offset = (ii / SubTile_H) * N_regs + (jj / SubTile_W);
        auto output_offset = (tile_row_start * M + ii + row_num) * n +
                             tile_col_start * N + threadIdx.x % 32 + jj;
        d[output_offset] =
            alpha * acc[acc_reg_offset] + beta * c[output_offset];
      }
    }
  }
}
}  // namespace nvidia::kernels

#endif
