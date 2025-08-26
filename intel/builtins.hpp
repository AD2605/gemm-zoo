#ifndef INTEL_BUILTINS_HPP
#define INTEL_BUILTINS_HPP

#include "cacheopts.hpp"
#include "defines.hpp"

#include <cstdint>
#include <sys/types.h>

// reads
SYCL_DEVICE_BUILTIN(
    intel::short8 __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(
        intptr_t baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::uint2 coord));
SYCL_DEVICE_BUILTIN(
    intel::uint8 __builtin_IB_subgroup_block_read_flat_u16_m16k16v1(
        intptr_t baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::uint2 coord));
SYCL_DEVICE_BUILTIN(
    intel::short8 __builtin_IB_subgroup_block_read_flat_u8_m8k32v1(
        intptr_t baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::uint2 coord));
SYCL_DEVICE_BUILTIN(
    intel::short16 __builtin_IB_subgroup_block_read_flat_u16_m8k32v1(
        intptr_t baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::uint2 coord));
SYCL_DEVICE_BUILTIN(
    intel::char32 __builtin_IB_subgroup_block_read_flat_u8_m32k16v1(
        intptr_t baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::uint2 coord));
SYCL_DEVICE_BUILTIN(
    intel::short32 __builtin_IB_subgroup_block_read_flat_u16_m32k16v1(
        intptr_t baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::uint2 coord));
SYCL_DEVICE_BUILTIN(
    intel::int8 __builtin_IB_subgroup_block_read_flat_u32_m8k16v1(
        intptr_t baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::uint2 coord));

// reads with transform
SYCL_DEVICE_BUILTIN(
    intel::int8 __builtin_IB_subgroup_block_read_flat_transform_u16_k16(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::uint2 coord));

// write
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u32_m8k16v1(
    intptr_t baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::uint2 coord, intel::uint8 data));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u16_m8k16v1(
    intptr_t baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::uint2 coord, intel::short8 data));

// prefetches
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::uint2 coord,
    intel::cacheopts::LSC_LDCC cacheOpt));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::uint2 coord,
    intel::cacheopts::LSC_LDCC cacheOpt));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u16_m8k32v1(
    intptr_t baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::uint2 coord,
    intel::cacheopts::LSC_LDCC cacheOpt));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v1(
    intptr_t baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::uint2 coord,
    intel::cacheopts::LSC_LDCC cacheOpt));

// prefetches with transform
SYCL_DEVICE_BUILTIN(
    void __builtin_IB_subgroup_block_read_prefetch_transform_u16_k16(
        long baseoffset, int width_minus_one, int height_minus_one,
        int pitch_minus_one, intel::uint2 coord,
        intel::cacheopts::LSC_LDCC cacheOpt));

// TODO: how to use these ? was getting undefined symbol during runtime
// SYCL_DEVICE_OCL(intel::float8 intel_sub_group_bf16_bf16_matrix_mad_k16(
//     intel::short8 a, intel::uint8 b, intel::float8 acc));

#endif
