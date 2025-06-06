cmake_minimum_required(VERSION 3.15)
project(gemm-zoo LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

set(TARGET_ARCH intel_gpu_bmg_g21 CACHE STRING "architecture to target")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/oneMath.cmake)

file(GLOB_RECURSE BENCHMARKS
    CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/bench/*.cpp
)

set(COMPILE_FLAGS "-fsycl;")
list(APPEND COMPILE_FLAGS "-fsycl-targets=${TARGET_ARCH};")
list(APPEND COMPILE_FLAGS "-Xspirv-translator;-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate")

foreach(BENCH_SOURCE ${BENCHMARKS})
    get_filename_component(BENCH_NAME ${BENCH_SOURCE} NAME_WE)

    add_executable(${BENCH_NAME} ${BENCH_SOURCE})
    add_dependencies(${BENCH_NAME} oneMath)
    target_compile_options(${BENCH_NAME} PUBLIC "${COMPILE_FLAGS}")
    target_link_options(${BENCH_NAME} PUBLIC "${COMPILE_FLAGS}")
    target_include_directories(${BENCH_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${ONEMATH_INSTALL_DIR}/include)
    target_link_libraries(${BENCH_NAME} PRIVATE ${ONEMATH_LIB})
endforeach()
