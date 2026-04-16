cmake_minimum_required(VERSION 3.22)

project(exp_20260416_cudss_analysis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(WITH_CUDA ON CACHE BOOL "Build cuPF CUDA backend" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable cuPF tests for experiment builds" FORCE)
set(BUILD_PYTHON_BINDINGS OFF CACHE BOOL "Disable Python bindings" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "Disable cuPF benchmark executables" FORCE)

find_package(CUDAToolkit REQUIRED)

get_filename_component(WORKSPACE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_subdirectory("${WORKSPACE_ROOT}/cuPF" "${CMAKE_CURRENT_BINARY_DIR}/cuPF")

add_executable(dump_cudss_reorder_data
  dump_cudss_reorder_data.cpp
  "${WORKSPACE_ROOT}/exp/20260409/cupf_dataset_loader.cpp"
  "${WORKSPACE_ROOT}/exp/20260409/powerflow_linear_system.cpp"
)

target_include_directories(dump_cudss_reorder_data
  PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${WORKSPACE_ROOT}/exp/20260409"
    "${WORKSPACE_ROOT}/cuPF/cpp/inc"
)

target_link_libraries(dump_cudss_reorder_data PRIVATE cupf)
