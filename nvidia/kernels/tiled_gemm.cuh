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
  for (; block_id < total_tiles; block_id += gridDim.x) {
    for (int i = 0; i < M_regs * N_regs; i++) {
      acc[i] = 0;
    }
    auto tile_row_start = block_id / N_tiles;
    auto tile_col_start = block_id % N_tiles;

    const int row_num = threadIdx.x / SubTile_H;

    for (int K_tile = 0; K_tile < k; K_tile += K) {
      // load a_tile;
#pragma unroll
      for (int i = 0; i < M; i += SubTile_H) {
#pragma unroll
        for (int j = 0; j < K; j += SubTile_W) {
          auto offset = (tile_row_start * M + row_num + i) * k + j +
                        threadIdx.x % 32 + K_tile;
          smem_a[(row_num + i) * K + j + threadIdx.x % 32] = a[offset];
        }
      }

      // load b_tile;
#pragma unroll
      for (int i = 0; i < K; i += SubTile_H) {
#pragma unroll
        for (int j = 0; j < N; j += SubTile_W) {
          auto offset = (K_tile + row_num + i) * n + j + threadIdx.x % 32 +
                        tile_col_start * N;
          smem_b[(row_num + i) * K + j + threadIdx.x % 32] = b[offset];
        }
      }

      __syncthreads();

      // if I have  M * N worth of data to be spread between 1024 threads, each
      // thread gets M * N / 1024; but since I also have to maintain coalesced
      // writes, they'll be spread with a weird stride of 32;
#pragma unroll
      for (int ii = 0; ii < M; ii += SubTile_H) {
        auto smem_a_row_offset = (ii + row_num) * K;
#pragma unroll
        for (int kk = 0; kk < K; kk++) {
          for (int jj = 0; jj < N; jj += SubTile_W) {
            const int acc_reg_offset =
                (ii / SubTile_H) * N_regs + (jj / SubTile_W);
            acc[acc_reg_offset] +=
                smem_a[smem_a_row_offset + threadIdx.x % 32 + jj] *
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
