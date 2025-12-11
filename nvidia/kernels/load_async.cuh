#ifndef NVIDIA_KERNELS_LOAD_ASYNC_CUH
#define NVIDIA_KERNELS_LOAD_ASYNC_CUH

#include "kernels/utils.cuh"

namespace nvidia::kernels::async_load {

template <typename T, int BM, int BK, int NumThreads>
__device__ __forceinline__ void load_swizzled(const T *gmem_ptr,
                                              const int32_t smem_addr,
                                              const int32_t &gmem_ld,
                                              const int32_t &gmem_row_offset,
                                              const int32_t &gmem_col_offset) {
  constexpr int elements_per_copy = 16 / sizeof(T);
  static_assert(BK % elements_per_copy == 0,
                "BK must be divisible by elements_per_copy");
  constexpr int chunks_per_row = BK / elements_per_copy;
  constexpr int total_chunks = BM * chunks_per_row;
  constexpr int NumBanks = 32;
  constexpr int BankSize = 4;
  constexpr int M = utils::log2_floor(elements_per_copy);
  constexpr int B = utils::log2_floor((NumBanks * BankSize) / sizeof(T)) - M;
  constexpr int S = utils::log2_floor(BK) - M;

  for (int i = threadIdx.x; i < total_chunks; i += NumThreads) {
    int row = i / chunks_per_row;
    int chunk_col = i % chunks_per_row;
    int col =
        chunk_col *
        elements_per_copy;  // Starting column (element index) for this chunk.
    int smem_logical = row * BK + col;
    int smem_swizzled = utils::swizzle<B, M, S>(smem_logical);
    int smem_swizzled_bytes = smem_swizzled * sizeof(T);
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16; \n" ::"r"(
                     smem_addr + smem_swizzled_bytes),
                 "l"(gmem_ptr + (row + gmem_row_offset) * gmem_ld +
                     gmem_col_offset + col));
  }
}

}  // namespace nvidia::kernels::async_load

#endif