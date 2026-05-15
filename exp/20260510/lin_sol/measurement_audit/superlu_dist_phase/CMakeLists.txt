cmake_minimum_required(VERSION 3.22)

project(superlu_dist_phase_diagnosis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP)

set(LIN_SOL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(COMMON_INCLUDE "${LIN_SOL_DIR}/common/cpp")
set(SUPERLU_DIST_PREFIX "${LIN_SOL_DIR}/third_party/superlu_dist/install")

find_library(SUPERLU_DIST_LIBRARY
    NAMES superlu_dist libsuperlu_dist.a
    PATHS "${SUPERLU_DIST_PREFIX}/lib"
    REQUIRED
)

find_path(SUPERLU_DIST_INCLUDE_DIR
    NAMES superlu_ddefs.h
    PATHS "${SUPERLU_DIST_PREFIX}/include"
    REQUIRED
)

add_executable(superlu_dist_phase_benchmark superlu_dist_phase_benchmark.cpp)
target_include_directories(superlu_dist_phase_benchmark PRIVATE
    "${COMMON_INCLUDE}"
    "${SUPERLU_DIST_INCLUDE_DIR}"
)
target_compile_definitions(superlu_dist_phase_benchmark PRIVATE
    SUPERLU_DIST_LIBRARY_PATH_STR="${SUPERLU_DIST_LIBRARY}"
)
target_link_libraries(superlu_dist_phase_benchmark PRIVATE
    MPI::MPI_CXX
    "${SUPERLU_DIST_LIBRARY}"
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    CUDA::cusparse
    OpenMP::OpenMP_CXX
    /usr/lib/x86_64-linux-gnu/libblas.so
    /usr/lib/x86_64-linux-gnu/liblapack.so
    m
)
