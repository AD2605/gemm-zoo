cmake_minimum_required(VERSION 3.15)
project(gemm-zoo)
set(CMAKE_CXX_STANDARD 17)

set(BUILD_FOLDER NVIDIA CACHE STRING "Folder to Build: Either NVIDIA or Intel")

if (BUILD_FOLDER STREQUAL "NVIDIA")
    message(STATUS "Chosen folder: NVIDIA")

    enable_language(CUDA)

    find_package(CUDAToolkit REQUIRED)
    
    set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "List of CUDA architectures to build for (e.g., 75;86)")

    file(GLOB_RECURSE BENCHMARKS
        CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/nvidia/bench/*.cu
    )

    set_source_files_properties(${BENCHMARKS} PROPERTIES LANGUAGE CUDA)
    
    set(COMPILE_FLAGS "")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND COMPILE_FLAGS "-G;-g;")
        add_compile_definitions(DEBUG)
    else()
        list(APPEND COMPILE_FLAGS "-lineinfo")
    endif()

    foreach(BENCH_SOURCE ${BENCHMARKS})
        get_filename_component(BENCH_NAME ${BENCH_SOURCE} NAME_WE)

        add_executable(${BENCH_NAME} ${BENCH_SOURCE})
        target_compile_options(${BENCH_NAME} PUBLIC "${COMPILE_FLAGS}")
        target_include_directories(${BENCH_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/nvidia)
        target_link_libraries(${BENCH_NAME} PRIVATE ${ONEMATH_LIB} CUDA::cublasLt)
        target_link_libraries(${BENCH_NAME} PRIVATE ${ONEMATH_LIB} CUDA::curand)
endforeach()

elseif(BUILD_FOLDER STREQUAL "INTEL")
    message(STATUS "Chosen folder: Intel")
    set(TARGET_ARCH intel_gpu_bmg_g21 CACHE STRING "architecture to target")

    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/oneMath.cmake)
    find_package(DNNL REQUIRED)

    file(GLOB_RECURSE BENCHMARKS
        CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/intel/bench/*.cpp
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
        target_include_directories(${BENCH_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/intel ${ONEMATH_INSTALL_DIR}/include)
        target_link_libraries(${BENCH_NAME} PRIVATE ${ONEMATH_LIB})
        target_link_libraries(${BENCH_NAME} PRIVATE ${ONEMATH_LIB} DNNL::dnnl)
endforeach()

else()
    message(FATAL_ERROR "BUILD_FOLDER should be either NVIDIA or INTEL")
endif()
