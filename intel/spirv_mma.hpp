#ifndef INTEL_SPIRV_MMA_HPP
#define INTEL_SPIRV_MMA_HPP

#include "defines.hpp"

struct SPIRV_MMAOperands {
  static constexpr int SPIRV_MatrixASigned = 0x1;
  static constexpr int SPIRV_MatrixBSigned = 0x2;
  static constexpr int SPIRV_MatrixAInt8 = 0x10;
  static constexpr int SPIRV_MatrixBInt8 = 0x20;
  static constexpr int SPIRV_MatrixAFp16 = 0x400;
  static constexpr int SPIRV_MatrixBFp16 = 0x800;
  static constexpr int SPIRV_MatrixABf16 = 0x1000;
  static constexpr int SPIRV_MatrixBBf16 = 0x2000;
  static constexpr int SPIRV_MatrixCBf16 = 0xC;
  static constexpr int SPIRV_MatrixATf32 = 0x100;
  static constexpr int SPIRV_MatrixBTf32 = 0x200;
};

SYCL_EXTERNAL intel::float8 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
    int32_t, intel::short8, intel::int8, intel::float8, int32_t);

#endif
