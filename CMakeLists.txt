cmake_minimum_required(VERSION 3.15)
project(gemm-zoo LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# 1) Include the file that creates the ExternalProject_Add(oneMath_project) target:
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/oneMath.cmake)

file(GLOB_RECURSE BENCHMARKS
     CONFIGURE_DEPENDS
     ${CMAKE_SOURCE_DIR}/bench/*.cpp
)

foreach(BENCH_SOURCE ${BENCHMARKS})
    get_filename_component(BENCH_NAME ${BENCH_SOURCE} NAME_WE)

    add_executable(${BENCH_NAME} ${BENCH_SOURCE})
    add_dependencies(${BENCH_NAME} oneMath)
    target_include_directories(${BENCH_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${ONEMATH_INSTALL_DIR}/include)
    target_link_libraries(${BENCH_NAME} PRIVATE ${ONEMATH_LIB})
endforeach()
