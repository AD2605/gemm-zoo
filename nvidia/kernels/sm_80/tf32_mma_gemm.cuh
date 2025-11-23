#ifndef NVIDIA_KERNELS_SM80_TF32_MMA_GEMM_CUH
#define NVIDIA_KERNELS_SM80_TF32_MMA_GEMM_CUH

#include "kernels/hilbert.cuh"
#include "kernels/morton_encoding.cuh"
#include "kernels/ring_buffer.cuh"
#include "kernels/utils.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace nvidia::kernels::sm80 {

namespace detail {

template <int LD>
__device__ __forceinline__ int swizzle_col(int y, int x) {
  constexpr int chunk_factor = 4;
  constexpr int actual_chunks = LD / chunk_factor;
  constexpr int num_chunks = []() constexpr {
    int nc = 1;
    while (nc * 2 <= actual_chunks) nc *= 2;
    return nc;
  }();
  constexpr int mask = num_chunks - 1;
  constexpr int log2n = []() constexpr {
    int l = 0;
    for (int t = num_chunks; t > 1; t /= 2) l++;
    return l;
  }();
  int i_chunk = y * actual_chunks + (x / chunk_factor);
  int y_chunk = i_chunk >> log2n;
  int x_chunk = i_chunk & mask;
  int x_chunk_swz = x_chunk ^ (y_chunk & mask);
  int x_swz = x_chunk_swz * chunk_factor + (x % chunk_factor);
  return x_swz;
}

template <int M, int K, int NumThreads, typename TIn>
__device__ __forceinline__ void async_populate_smemA_buffer(
    const int32_t smem_a_addr, const TIn* gmem_ptr, const int tile_row_start,
    const int kk, int k) {
  constexpr int vec_size = 16 / sizeof(TIn);
  constexpr int total_elements = (M * K) / vec_size;
  const uint32_t row_offset = tile_row_start * M;
  for (int i = threadIdx.x; i < total_elements; i += NumThreads) {
    int row_id = i / (K / vec_size);
    int col_id = (i % (K / vec_size)) * vec_size;
    int col_swz = detail::swizzle_col<K>(row_id, col_id);
    uint32_t smem_offset =
        (row_id * K + col_swz) * static_cast<int32_t>(sizeof(TIn));
    uint32_t gmem_offset = (row_offset + row_id) * k + (kk + col_id);
    asm volatile(
        "{\n\t"
        "cp.async.cg.shared.global.L2::256B [%0], [%1], 16; \n\t"
        "}"
        :
        : "r"(smem_a_addr + smem_offset), "l"(gmem_ptr + gmem_offset)
        : "memory");
  }
}

template <int K, int N, int NumThreads, typename TIn>
__device__ __forceinline__ void async_populate_smemB_buffer(
    const int32_t smem_b_addr, const TIn* gmem_ptr, const int tile_col_start,
    const int kk, int n) {
  constexpr int vec_size = 16 / sizeof(TIn);
  constexpr int total_elements = (N * K) / vec_size;
  const int col_offset = tile_col_start * N;
  for (int i = threadIdx.x; i < total_elements; i += NumThreads) {
    int row_id = i / (N / vec_size);
    int col_id = (i % (N / vec_size)) * (16 / sizeof(TIn));
    int col_swz = detail::swizzle_col<N>(row_id, col_id);
    uint32_t smem_offset =
        (row_id * N + col_swz) * static_cast<int32_t>(sizeof(TIn));
    uint32_t gmem_offset = (kk + row_id) * n + (col_offset + col_id);
    asm volatile(
        "{\n\t"
        "cp.async.cg.shared.global.L2::256B [%0], [%1], 16; \n\t"
        "}"
        :
        : "r"(smem_b_addr + smem_offset), "l"(gmem_ptr + gmem_offset)
        : "memory");
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

  const int M_tiles = utils::ceil_div(m, M);
  const int N_tiles = utils::ceil_div(n, N);
  const int total_tiles = M_tiles * N_tiles;

  auto t_idx = threadIdx.x % 32;

  int group_id = t_idx >> 2;
  int thread_id_in_group = t_idx % 4;
  constexpr int NumWarps = BlockDim / 32;

  constexpr int WM = 32;
  constexpr int WN = 64;
  constexpr int WK = 16;

  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 8;

  constexpr int TM = 4;
  constexpr int TN = 2;
  constexpr int TC = 4;

  const int warp_id = threadIdx.x / 32;
  constexpr int num_warps_per_row = N / WN;
  auto warp_tile_row_id = warp_id / num_warps_per_row;
  auto warp_tile_col_id = warp_id % num_warps_per_row;

  const int smem_load_a_offset = warp_tile_row_id * WM + group_id;
  const int smem_load_b_offset = warp_tile_col_id * WN + group_id;

  uint32_t tile_row_start;
  uint32_t tile_col_start;

  uint32_t tile_row_start_temp;
  uint32_t tile_col_start_temp;

  int head = 0;
  int tail = 0;

  utils::map_to_hilbert(block_id, N_tiles, tile_col_start, tile_row_start);

  // start the transfer of all the buffers;
  int k_load_index = 0;
#pragma unroll(NumBuffers)
  for (int i = 0; i < NumBuffers; i++) {
    if (k_load_index < k) {
      detail::async_populate_smemA_buffer<M, K, BlockDim, TIn>(
          smem_a_addr + tail * M * K * sizeof_TIn, a, tile_row_start,
          k_load_index, k);
      detail::async_populate_smemB_buffer<K, N, BlockDim, TIn>(
          smem_b_addr + tail * K * N * sizeof_TIn, b, tile_col_start,
          k_load_index, n);
      asm volatile("cp.async.commit_group;\n");
      k_load_index += K;
      tail = (tail + 1) % NumBuffers;
    }
  }

  while (block_id < total_tiles) {
    TOut C_regs[WM / MMA_M][WN / MMA_N][TC];  // 4 x 4 x 4
    for (int _i = 0; _i < WM / MMA_M; _i++) {
      for (int _j = 0; _j < WN / MMA_N; _j++) {
        for (int _k = 0; _k < TC; _k++) {
          C_regs[_i][_j][_k] = 0;
        }
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
            int a_row0 = smem_load_a_offset + i * MMA_M;
            int a_row1 = a_row0 + 8;
            int a_col0 = inner + j * MMA_K + thread_id_in_group;
            int a_col1 = a_col0 + 4;
            A_regs[j][i][0] =
                smem_a_ptr[a_row0 * K + detail::swizzle_col<K>(a_row0, a_col0)];
            A_regs[j][i][2] =
                smem_a_ptr[a_row0 * K + detail::swizzle_col<K>(a_row0, a_col1)];
            A_regs[j][i][1] =
                smem_a_ptr[a_row1 * K + detail::swizzle_col<K>(a_row1, a_col0)];
            A_regs[j][i][3] =
                smem_a_ptr[a_row1 * K + detail::swizzle_col<K>(a_row1, a_col1)];
          }
        }

#pragma unroll
        for (int i = 0; i < WK / MMA_K; i++) {
#pragma unroll
          for (int j = 0; j < WN / MMA_N; j++) {
            int b_row0 = inner + i * MMA_K + thread_id_in_group;
            int b_row1 = b_row0 + 4;
            int b_col = smem_load_b_offset + j * MMA_N;
            B_regs[i][j][0] =
                smem_b_ptr[b_row0 * N + detail::swizzle_col<N>(b_row0, b_col)];
            B_regs[i][j][1] =
                smem_b_ptr[b_row1 * N + detail::swizzle_col<N>(b_row1, b_col)];
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

      head = (head + 1) % NumBuffers;

      // populate the (k + 3)th buffer;
      if (k_load_index < k) {
        detail::async_populate_smemA_buffer<M, K, BlockDim, TIn>(
            smem_a_addr + tail * M * K * sizeof_TIn, a, tile_row_start,
            k_load_index, k);
        detail::async_populate_smemB_buffer<K, N, BlockDim, TIn>(
            smem_b_addr + tail * K * N * sizeof_TIn, b, tile_col_start,
            k_load_index, n);
        asm volatile("cp.async.commit_group;\n");
        k_load_index += K;
        tail = (tail + 1) % NumBuffers;
      }
    }

    __syncthreads();
    // Pipeline loads from global Memory;
    // Output Tile Size M x N;
    int32_t d_row_offset = tile_row_start * M + warp_tile_row_id * WM;
    int32_t d_col_offset =
        tile_col_start * N + warp_tile_col_id * WN + 2 * t_idx;
    TOut D_regs[WM][WN / 32];
#pragma unroll
    for (int d_row = 0; d_row < WM; d_row++) {
#pragma unroll
      for (int d_col = 0; d_col < WN; d_col += 64) {
        asm volatile(
            "ld.global.cs.v2.f32 {%0, %1}, [%2]; \n\t"
            : "=f"(D_regs[d_row][d_col / 32 + 0]),
              "=f"(D_regs[d_row][d_col / 32 + 1])
            : "l"(&c[(d_row_offset + d_row) * n + d_col_offset + d_col]));
      }
    }

    block_id += gridDim.x;
    k_load_index = 0;
    head = 0;
    tail = 0;
    // pipeline loads to smem for next tile;
    utils::map_to_hilbert(block_id, N_tiles, tile_col_start_temp,
                          tile_row_start_temp);
#pragma unroll(NumBuffers)
    for (int i = 0; i < NumBuffers; i++) {
      if (k_load_index < k) {
        detail::async_populate_smemA_buffer<M, K, BlockDim, TIn>(
            smem_a_addr + tail * M * K * sizeof_TIn, a, tile_row_start_temp,
            k_load_index, k);
        detail::async_populate_smemB_buffer<K, N, BlockDim, TIn>(
            smem_b_addr + tail * K * N * sizeof_TIn, b, tile_col_start_temp,
            k_load_index, n);
        asm volatile("cp.async.commit_group;\n");
        k_load_index += K;
        tail = (tail + 1) % NumBuffers;
      }
    }

    // so each warp now writes WN to shared memory, and then writes to the
    // output
    int32_t smem_output_buffer_addr =
        smem_a_addr +
        NumBuffers * static_cast<int32_t>(sizeof(TIn)) * (M * K + K * N) +
        warp_id * WN * sizeof(TOut);
    // as each WM x WN is made up of 16 x 8 matrices, I will have 2 matrices
    // along rows and 4 matrices along columns (as WM is hardcoded to 32 and WN
    // is hardcoded to 64)

    // TODO: implement;
    // GROK: you would need to lookup the tf32 mma C matrix mapping
#pragma unroll
    for (int d_row = 0; d_row < WM; d_row++) {
      const int num_matrix = d_row / 16;
      const int resp_thread_id_start = (d_row % 8) * 4;
      const int reg_id_offset = (d_row % 16) / 8;
      const int output_smem_thread_offset =
          smem_output_buffer_addr + 2 * (t_idx - resp_thread_id_start) *
                                        static_cast<int32_t>(sizeof(TOut));
#pragma unroll
      for (int d_col = 0; d_col < WN; d_col += MMA_N) {
        if (t_idx >= resp_thread_id_start && t_idx < resp_thread_id_start + 4) {
          asm volatile(
              "st.shared.v2.f32 [%0], {%1, %2}; \n\t"
              :
              : "r"(output_smem_thread_offset +
                    8 * (d_col / MMA_N) * static_cast<int32_t>(sizeof(TOut))),
                "f"(C_regs[d_row / MMA_M][d_col / MMA_N][2 * reg_id_offset +
                                                         0]),
                "f"(C_regs[d_row / MMA_M][d_col / MMA_N][2 * reg_id_offset + 1])
              : "memory");
        }
      }
      __syncwarp();
      for (int d_col = 0; d_col < WN; d_col += 64) {
        float v01, v02;
        asm volatile("ld.shared.v2.f32 {%0, %1}, [%2]; \n\t"
                     : "=f"(v01), "=f"(v02)
                     : "r"(smem_output_buffer_addr +
                           2 * t_idx * static_cast<int32_t>(sizeof(TOut))));
        D_regs[d_row][d_col / 32 + 0] *= beta;
        D_regs[d_row][d_col / 32 + 1] *= beta;
        D_regs[d_row][d_col / 32 + 0] += alpha * v01;
        D_regs[d_row][d_col / 32 + 1] += alpha * v02;
        asm volatile("st.global.cs.v2.f32 [%0], {%1, %2}; \n\t" ::"l"(
                         &d[(d_row_offset + d_row) * n + d_col_offset + d_col]),
                     "f"(D_regs[d_row][d_col / 32 + 0]),
                     "f"(D_regs[d_row][d_col / 32 + 1])
                     : "memory");
      }
      __syncwarp();
    }
    tile_col_start = tile_col_start_temp;
    tile_row_start = tile_row_start_temp;
  }
}
}  // namespace nvidia::kernels::sm80

#endif
