cmake_minimum_required(VERSION 3.22)

project(exp_20260519_reordering LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

find_library(NVTOOLSEXT_LIBRARY NAMES nvToolsExt
  PATHS
    /usr/local/cuda/lib64
    /usr/local/cuda/targets/x86_64-linux/lib
    /usr/local/cuda-12.8/targets/x86_64-linux/lib
    /usr/lib
    /usr/lib/x86_64-linux-gnu)
if(NOT NVTOOLSEXT_LIBRARY)
  message(FATAL_ERROR "NVTX library nvToolsExt is required")
endif()

find_library(METIS_LIBRARY NAMES metis libmetis.so.5
  PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
if(NOT METIS_LIBRARY)
  message(FATAL_ERROR "METIS library is required")
endif()

find_path(CUDSS_INCLUDE_DIR cudss.h
  PATHS
    /root/.local/lib/python3.10/site-packages/nvidia/cu12/include
    /usr/local/cuda/include
    /usr/include)
find_library(CUDSS_LIBRARY NAMES cudss libcudss.so
  PATHS
    /root/.local/lib/python3.10/site-packages/nvidia/cu12/lib
    /usr/local/cuda/lib64
    /usr/lib
    /usr/lib/x86_64-linux-gnu)
if(NOT CUDSS_INCLUDE_DIR OR NOT CUDSS_LIBRARY)
  message(FATAL_ERROR "cuDSS headers and library are required")
endif()

add_library(exp20260519_core STATIC
  src/io/matrix_market_io.c
  src/matrix/csc_matrix.c
  src/matrix/csr_matrix.c
  src/numeric/numeric_cuda_runtime.cu
  src/numeric/numeric_factorization_cuda.cu
  src/numeric/numeric_front_dense_cuda.cu
  src/numeric/numeric_kernels_cuda.cu
  src/numeric/numeric_plan_cuda.cu
  src/reordering/metis_nd.c
  src/symbolic/etree.c
  src/symbolic/assembly_plan.c
  src/symbolic/front_schedule.c
  src/symbolic/front_symbolic.c
  src/symbolic/septree.c
  src/symbolic/symbolic_factorization.c
  src/symbolic/symbolic_internal.c
  src/symbolic/symbolic_print.c
  src/symbolic/symbolic_validate.c
  src/symbolic/storage_plan.c
)

target_include_directories(exp20260519_core
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/io"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/matrix"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/numeric"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/reordering"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/symbolic"
)

target_link_libraries(exp20260519_core
  PUBLIC
    "${METIS_LIBRARY}"
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)

add_executable(custom
  benchmark/custom.c
)

target_link_libraries(custom
  PRIVATE
    exp20260519_core
)

add_executable(cudss_pf_benchmark
  benchmark/cudss_pf_benchmark.cpp
)

target_include_directories(cudss_pf_benchmark
  PRIVATE
    "${CUDSS_INCLUDE_DIR}"
)

target_link_libraries(cudss_pf_benchmark
  PRIVATE
    exp20260519_core
    CUDA::cudart
    "${NVTOOLSEXT_LIBRARY}"
    "${CUDSS_LIBRARY}"
)

get_filename_component(CUDSS_LIBRARY_DIR "${CUDSS_LIBRARY}" DIRECTORY)
set_target_properties(cudss_pf_benchmark PROPERTIES
  BUILD_RPATH "${CUDSS_LIBRARY_DIR}"
)

enable_testing()

add_executable(test_ordering
  tests/test_ordering.c
)

target_link_libraries(test_ordering
  PRIVATE
    exp20260519_core
    m
)

add_test(NAME ordering_tests COMMAND test_ordering)

add_executable(test_symbolic
  tests/test_symbolic.c
)

target_link_libraries(test_symbolic
  PRIVATE
    exp20260519_core
    m
)

add_test(NAME symbolic_tests COMMAND test_symbolic)

add_executable(test_numeric_cuda
  tests/test_numeric_cuda.c
)

target_link_libraries(test_numeric_cuda
  PRIVATE
    exp20260519_core
    m
)

add_test(NAME numeric_cuda_tests COMMAND test_numeric_cuda)
