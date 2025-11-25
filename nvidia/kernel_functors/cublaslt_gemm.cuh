#ifndef CUBLASLT_GEMM_CUH
#define CUBLASLT_GEMM_CUH

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace nvidia::kernel_functors {
template <typename TIn, typename TOut>
struct cublasLt_gemm {
  cublasLt_gemm(std::size_t m, std::size_t n, std::size_t k,
                const cudaDeviceProp& properties)
      : m(m), n(n), k(k) {
    cublasStatus_t status = cublasLtCreate(&ltHandle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to create cuBLASLt handle");
    }

    cublasLtMatrixLayoutCreate(&layoutA, get_cubaslt_datatype<TIn>(), m, k, k);
    cublasLtMatrixLayoutCreate(&layoutB, get_cubaslt_datatype<TIn>(), k, n, n);
    cublasLtMatrixLayoutCreate(&layoutC, get_cubaslt_datatype<TOut>(), m, n, n);
    cublasLtMatrixLayoutCreate(&layoutD, get_cubaslt_datatype<TOut>(), m, n, n);

    cublasComputeType_t computeType = get_compute_type<TIn>();
    cublasLtMatmulDescCreate(&matmulDesc, computeType, CUDA_R_32F);

    cublasOperation_t opTranspose = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                   &opTranspose, sizeof(opTranspose));
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                   &opTranspose, sizeof(opTranspose));

    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                   &epilogue, sizeof(epilogue));

    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);

    cublasLtMatmulHeuristicResult_t heuristicResult;
    int returnedResults = 0;
    status = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc, layoutA, layoutB, layoutC, layoutD, preference, 1,
        &heuristicResult, &returnedResults);
    if (status != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
      throw std::runtime_error("No suitable cuBLASLt algorithm found");
    }

    algo = heuristicResult.algo;

    workspaceSize = heuristicResult.workspaceSize;
    if (workspaceSize > 0) {
      cudaError_t cudaStat = cudaMalloc(&workspace, workspaceSize);
      if (cudaStat != cudaSuccess) {
        throw std::runtime_error("Failed to allocate workspace");
      }
    }
  }

  void operator()(const TIn* a, const TIn* b, const TOut* c, TOut* d,
                  const TOut alpha, const TOut beta, cudaStream_t stream) {
    cublasLtMatmul(ltHandle, matmulDesc, &alpha, a, layoutA, b, layoutB, &beta,
                   c, layoutC, d, layoutD, &algo, workspace, workspaceSize,
                   stream);
  }

 private:
  template <typename T>
  cudaDataType_t get_cubaslt_datatype() {
    if constexpr (std::is_same_v<T, float>) {
      return CUDA_R_32F;
    }
    if constexpr (std::is_same_v<T, __half>) {
      return CUDA_R_16F;
    }
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      return CUDA_R_16BF;
    }
    throw std::runtime_error("Unsupported Datatype");
  }

  template <typename T>
  auto get_compute_type() {
    if constexpr (std::is_same_v<T, float>) {
      return CUBLAS_COMPUTE_32F;
    }
    if constexpr (std::is_same_v<T, __half>) {
      return CUDA_R_16F;
    }
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      return CUDA_R_16BF;
    }
    throw std::runtime_error("Unsupported Datatype");
  }

  cublasLtHandle_t ltHandle;
  void* workspace = nullptr;
  std::size_t workspaceSize;
  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatrixLayout_t layoutA;
  cublasLtMatrixLayout_t layoutB;
  cublasLtMatrixLayout_t layoutC;
  cublasLtMatrixLayout_t layoutD;
  cublasLtMatmulAlgo_t algo;
  std::size_t m;
  std::size_t n;
  std::size_t k;
};
}  // namespace nvidia::kernel_functors

#endif
