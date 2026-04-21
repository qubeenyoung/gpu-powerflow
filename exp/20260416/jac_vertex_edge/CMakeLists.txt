cmake_minimum_required(VERSION 3.22)

project(exp_20260416_jac_vertex_edge LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)
endif()

set(WITH_CUDA ON CACHE BOOL "Build cuPF CUDA backend" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable cuPF tests for experiment builds" FORCE)
set(BUILD_PYTHON_BINDINGS OFF CACHE BOOL "Disable Python bindings" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "Disable cuPF benchmark executables" FORCE)
set(ENABLE_LOG OFF CACHE BOOL "Disable cuPF logging" FORCE)
set(ENABLE_DUMP OFF CACHE BOOL "Disable cuPF dump utilities" FORCE)
set(ENABLE_TIMING OFF CACHE BOOL "Disable cuPF scoped timing for event probe" FORCE)
set(ENABLE_NVTX ON CACHE BOOL "Enable cuPF NVTX when available" FORCE)

find_package(CUDAToolkit REQUIRED)

get_filename_component(WORKSPACE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_subdirectory("${WORKSPACE_ROOT}/cuPF" "${CMAKE_CURRENT_BINARY_DIR}/cuPF")

add_executable(jacobian_operator_probe
  jacobian_operator_probe.cu
  "${WORKSPACE_ROOT}/cuPF/tests/dump_case_loader.cpp"
)

target_include_directories(jacobian_operator_probe
  PRIVATE
    "${WORKSPACE_ROOT}/cuPF/cpp/inc"
    "${WORKSPACE_ROOT}/cuPF/cpp/src"
    "${WORKSPACE_ROOT}/cuPF/tests"
)

target_link_libraries(jacobian_operator_probe
  PRIVATE
    cupf
    CUDA::cudart
)

set_target_properties(jacobian_operator_probe PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
