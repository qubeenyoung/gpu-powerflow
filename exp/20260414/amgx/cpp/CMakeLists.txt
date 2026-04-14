cmake_minimum_required(VERSION 3.22)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(exp_20260414_amgx_cpp LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(WITH_CUDA ON CACHE BOOL "Build cuPF CUDA backend" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "Disable cuPF benchmarks for this experiment" FORCE)
set(BUILD_PYTHON_BINDINGS OFF CACHE BOOL "Disable cuPF Python bindings for this experiment" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable cuPF tests for this experiment" FORCE)
set(ENABLE_LOG OFF CACHE BOOL "Disable cuPF logging for this experiment" FORCE)
set(ENABLE_DUMP OFF CACHE BOOL "Disable cuPF debug dumps for this experiment" FORCE)
set(ENABLE_TIMING OFF CACHE BOOL "Disable cuPF internal timing for this experiment" FORCE)

find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

find_path(AMGX_INCLUDE_DIR amgx_c.h PATHS /usr/local/include)
find_library(AMGX_LIBRARY NAMES amgxsh amgx PATHS /usr/local/lib)
if(NOT AMGX_INCLUDE_DIR OR NOT AMGX_LIBRARY)
  message(FATAL_ERROR "AMGX was not found")
endif()

get_filename_component(WORKSPACE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(CUPF_ROOT "${WORKSPACE_ROOT}/cuPF")

add_subdirectory("${CUPF_ROOT}" "${CMAKE_BINARY_DIR}/cuPF")

add_executable(amgx_jfnk_probe
  jfnk_amgx_gpu.cu
  newton_krylov_probe.cpp
  "${CUPF_ROOT}/tests/dump_case_loader.cpp"
)

target_include_directories(amgx_jfnk_probe
  PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${CUPF_ROOT}/cpp/inc"
    "${CUPF_ROOT}/cpp/src"
    "${CUPF_ROOT}/tests"
    "${AMGX_INCLUDE_DIR}"
)

target_link_libraries(amgx_jfnk_probe
  PRIVATE
    cupf
    Eigen3::Eigen
    CUDA::cudart
    CUDA::cublas
    OpenMP::OpenMP_CXX
    "${AMGX_LIBRARY}"
)

set_target_properties(amgx_jfnk_probe PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "/usr/local/lib"
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES 86
)
