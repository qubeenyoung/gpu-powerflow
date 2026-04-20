cmake_minimum_required(VERSION 3.22)

project(exp20260420_linear_solver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

find_path(CUDSS_INCLUDE_DIR
  NAMES cudss.h cuDSS.h
  HINTS
    /usr/include
    /usr/local/include
    /usr/local/cuda/include
    $ENV{HOME}/.local/lib/python3.10/site-packages/nvidia/cu12/include)
find_library(CUDSS_LIBRARY
  NAMES cudss
  HINTS
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/lib
    $ENV{HOME}/.local/lib/python3.10/site-packages/nvidia/cu12/lib)

if(NOT CUDSS_INCLUDE_DIR OR NOT CUDSS_LIBRARY)
  message(FATAL_ERROR "cuDSS headers/library are required for exp20260420_linear_solver")
endif()

add_library(exp20260420_linear_solver STATIC
  src/cudss_lu.cu
)

target_include_directories(exp20260420_linear_solver PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CUDSS_INCLUDE_DIR}
)

target_link_libraries(exp20260420_linear_solver PUBLIC
  CUDA::cudart
  ${CUDSS_LIBRARY}
)

set_target_properties(exp20260420_linear_solver PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
