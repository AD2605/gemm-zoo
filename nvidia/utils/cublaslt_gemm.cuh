#ifndef NVIDIA_UTILS_CUBLASLT_GEMM_CUH
#define NVIDIA_UTILS_CUBLASLT_GEMM_CUH

#include "defines.hpp"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace utils {

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
  if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
    return CUDA_R_8F_E4M3;
  }
  if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
    return CUDA_R_8F_E5M2;
  }
  throw std::runtime_error("Unsupported Datatype");
}

template <typename T>
auto get_compute_type() {
  if constexpr (std::is_same_v<T, float>) {
    return CUBLAS_COMPUTE_32F_FAST_TF32;
  }
  if constexpr (std::is_same_v<T, __half>) {
    return CUBLAS_COMPUTE_16F;
    ;
  }
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return CUBLAS_COMPUTE_32F_FAST_16BF;
  }
  if constexpr (std::is_same_v<T, __nv_fp8_e4m3> ||
                std::is_same_v<T, __nv_fp8_e5m2>) {
    return CUBLAS_COMPUTE_16F;  // GROK PLEASE CHECK IF THIS IS EVEN CORRECT
  }
  return CUBLAS_COMPUTE_32F;
}

template <typename TA, typename TB, typename TC, typename TD>
void cublaslt_gemm(const TA* a, const TB* b, const TC* c, TD* d, int m, int n,
                   int k, TD alpha, TD beta, cudaStream_t stream) {
  cublasLtHandle_t ltHandle;
  cublasStatus_t status = cublasLtCreate(&ltHandle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to create cuBLASLt handle");
  }

  cublasLtMatrixLayout_t layoutA, layoutB, layoutC, layoutD;
  cublasLtMatrixLayoutCreate(&layoutA, utils::get_cubaslt_datatype<TA>(), m, k,
                             k);
  cublasLtMatrixLayoutCreate(&layoutB, utils::get_cubaslt_datatype<TB>(), k, n,
                             n);
  cublasLtMatrixLayoutCreate(&layoutC, utils::get_cubaslt_datatype<TC>(), m, n,
                             n);
  cublasLtMatrixLayoutCreate(&layoutD, utils::get_cubaslt_datatype<TD>(), m, n,
                             n);

  cublasLtMatmulDesc_t matmulDesc;
  cublasComputeType_t computeType = get_compute_type<TA>();
  cublasLtMatmulDescCreate(&matmulDesc, computeType,
                           get_cubaslt_datatype<TD>());

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

  cublasLtMatmulAlgo_t algo = heuristicResult.algo;

  void* workspace = nullptr;
  size_t workspaceSize = heuristicResult.workspaceSize;
  if (workspaceSize > 0) {
    cudaError_t cudaStat = cudaMalloc(&workspace, workspaceSize);
    if (cudaStat != cudaSuccess) {
      throw std::runtime_error("Failed to allocate workspace");
    }
  }

  status = cublasLtMatmul(ltHandle, matmulDesc, &alpha, a, layoutA, b, layoutB,
                          &beta, c, layoutC, d, layoutD, &algo, workspace,
                          workspaceSize, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    if (workspace) cudaFree(workspace);
    throw std::runtime_error("cublasLtMatmul failed");
  }

  checkCudaError(cudaStreamSynchronize(stream));

  if (workspace) {
    cudaFree(workspace);
  }
  cublasLtMatmulPreferenceDestroy(preference);
  cublasLtMatmulDescDestroy(matmulDesc);
  cublasLtMatrixLayoutDestroy(layoutA);
  cublasLtMatrixLayoutDestroy(layoutB);
  cublasLtMatrixLayoutDestroy(layoutC);
  cublasLtMatrixLayoutDestroy(layoutD);
  cublasLtDestroy(ltHandle);
}

}  // namespace utils
#endif
