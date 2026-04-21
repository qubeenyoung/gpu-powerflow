cmake_minimum_required(VERSION 3.22)

project(exp20260420_newton_solver LANGUAGES CXX CUDA)

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
  message(FATAL_ERROR "cuDSS headers/library are required for exp20260420_newton_solver")
endif()

add_library(exp20260420_newton_solver STATIC
  src/newton_solver.cu
  ../mismatch/src/mismatch.cu
  ../linear_solver/src/cudss_lu.cu
  ../update_voltage/src/update_voltage.cu
  ../jac_asm/src/common/jacobian_build.cpp
  ../jac_asm/src/edge/edge_fill_jac.cu
)

target_include_directories(exp20260420_newton_solver PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/../mismatch/src
  ${CMAKE_CURRENT_SOURCE_DIR}/../linear_solver/src
  ${CMAKE_CURRENT_SOURCE_DIR}/../update_voltage/src
  ${CMAKE_CURRENT_SOURCE_DIR}/../jac_asm/src/common
  ${CMAKE_CURRENT_SOURCE_DIR}/../jac_asm/src/edge
  ${CUDSS_INCLUDE_DIR}
)

target_link_libraries(exp20260420_newton_solver PUBLIC
  CUDA::cudart
  CUDA::cusparse
  ${CUDSS_LIBRARY}
)

set_target_properties(exp20260420_newton_solver PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

add_executable(exp20260420_newton_solver_probe
  src/newton_solver_probe.cu
)

target_link_libraries(exp20260420_newton_solver_probe PRIVATE
  exp20260420_newton_solver
)

set_target_properties(exp20260420_newton_solver_probe PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
