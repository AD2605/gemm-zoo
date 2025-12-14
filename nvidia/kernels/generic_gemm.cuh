#ifndef NVIDIA_KERNELS_GENERIC_GEMM_CUH
#define NVIDIA_KERNELS_GENERIC_GEMM_CUH

namespace nvidia::kernels {
template <typename TIn, typename TOut>
__global__ void generic_gemm(const TIn* a, const TIn* b, const TOut* c, TOut* d,
                             TOut alpha, TOut beta, int m, int n, int k,
                             int lda, int ldb, int ldc, int ldd) {
  // Basic GEMM with hardcoded tiling parameters to serve as a fallback in case
  // cublasLT heuristics fail and provide reference output.
  // This accepts any datatype, does all the compute
  // in float. Performance is not a consideration here (but the runtime should
  // be decent enough) should work on all types (fp8, bf/fp16 and fp32)
  constexpr int TILE_WIDTH = 32;

  __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
  __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  float sum = 0.0f;

  for (int p = 0; p < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++p) {
    // Load A tile
    if (row < m && (p * TILE_WIDTH + tx) < k) {
      A_tile[ty][tx] = static_cast<float>(a[row * lda + (p * TILE_WIDTH + tx)]);
    } else {
      A_tile[ty][tx] = 0.0f;
    }

    // Load B tile
    if ((p * TILE_WIDTH + ty) < k && col < n) {
      B_tile[ty][tx] = static_cast<float>(b[(p * TILE_WIDTH + ty) * ldb + col]);
    } else {
      B_tile[ty][tx] = 0.0f;
    }

    __syncthreads();

    for (int q = 0; q < TILE_WIDTH; ++q) {
      sum += A_tile[ty][q] * B_tile[q][tx];
    }

    __syncthreads();
  }

  if (row < m && col < n) {
    TOut val = static_cast<TOut>(sum) * alpha;
    if (beta != static_cast<TOut>(0)) {
      val += beta * c[row * ldc + col];
    }
    d[row * ldd + col] = val;
  }
}
}  // namespace nvidia::kernels

#endif
