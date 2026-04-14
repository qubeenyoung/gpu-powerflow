cmake_minimum_required(VERSION 3.22)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(exp_20260414_newton_krylov_cpp LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(WITH_CUDA OFF CACHE BOOL "Build cuPF CUDA backend" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "Disable cuPF benchmarks for this experiment" FORCE)
set(BUILD_PYTHON_BINDINGS OFF CACHE BOOL "Disable cuPF Python bindings for this experiment" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable cuPF tests for this experiment" FORCE)
set(ENABLE_LOG OFF CACHE BOOL "Disable cuPF logging for this experiment" FORCE)
set(ENABLE_DUMP OFF CACHE BOOL "Disable cuPF debug dumps for this experiment" FORCE)
set(ENABLE_TIMING OFF CACHE BOOL "Disable cuPF internal timing for this experiment" FORCE)

find_package(Eigen3 REQUIRED)

option(NEWTON_KRYLOV_WITH_HYPRE "Build JFNK HYPRE BoomerAMG preconditioner" ON)
if(NEWTON_KRYLOV_WITH_HYPRE)
  find_package(MPI REQUIRED)
  find_path(HYPRE_INCLUDE_DIR HYPRE.h PATH_SUFFIXES hypre)
  find_library(HYPRE_LIBRARY NAMES HYPRE)
  if(NOT HYPRE_INCLUDE_DIR OR NOT HYPRE_LIBRARY)
    message(FATAL_ERROR "NEWTON_KRYLOV_WITH_HYPRE=ON but HYPRE was not found")
  endif()
endif()

get_filename_component(WORKSPACE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(CUPF_ROOT "${WORKSPACE_ROOT}/cuPF")

add_subdirectory("${CUPF_ROOT}" "${CMAKE_BINARY_DIR}/cuPF")

add_executable(newton_krylov_probe
  jfnk_bicgstab.cpp
  newton_krylov_probe.cpp
  "${CUPF_ROOT}/tests/dump_case_loader.cpp"
)

target_include_directories(newton_krylov_probe
  PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${CUPF_ROOT}/cpp/inc"
    "${CUPF_ROOT}/cpp/src"
    "${CUPF_ROOT}/tests"
)

target_link_libraries(newton_krylov_probe
  PRIVATE
    cupf
    Eigen3::Eigen
)

if(NEWTON_KRYLOV_WITH_HYPRE)
  target_compile_definitions(newton_krylov_probe PRIVATE NEWTON_KRYLOV_WITH_HYPRE=1)
  target_include_directories(newton_krylov_probe PRIVATE "${HYPRE_INCLUDE_DIR}")
  target_link_libraries(newton_krylov_probe PRIVATE "${HYPRE_LIBRARY}" MPI::MPI_CXX)
endif()

set_target_properties(newton_krylov_probe PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
