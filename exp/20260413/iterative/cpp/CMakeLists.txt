cmake_minimum_required(VERSION 3.22)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(exp_20260413_iterative_cpp LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ITERATIVE_WITH_CUDA "Build the dump tool with cuPF CUDA/cuDSS profiles" ON)
if(ITERATIVE_WITH_CUDA)
  enable_language(CUDA)
endif()

set(WITH_CUDA ${ITERATIVE_WITH_CUDA} CACHE BOOL "Build cuPF CUDA backend" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "Disable cuPF benchmarks for this experiment" FORCE)
set(BUILD_PYTHON_BINDINGS OFF CACHE BOOL "Disable cuPF Python bindings for this experiment" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable cuPF tests for this experiment" FORCE)
set(ENABLE_LOG OFF CACHE BOOL "Disable cuPF logging for this experiment" FORCE)
set(ENABLE_TIMING ON CACHE BOOL "Enable cuPF timing/synchronization for this experiment" FORCE)

find_package(Eigen3 REQUIRED)

get_filename_component(WORKSPACE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(CUPF_ROOT "${WORKSPACE_ROOT}/cuPF")

add_subdirectory("${CUPF_ROOT}" "${CMAKE_BINARY_DIR}/cuPF")

add_library(iterative_linear_system_io STATIC
  linear_system_io.cpp
)

target_include_directories(iterative_linear_system_io
  PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}"
)

target_link_libraries(iterative_linear_system_io PUBLIC Eigen3::Eigen)

add_executable(dump_linear_systems
  dump_linear_systems.cpp
  "${CUPF_ROOT}/tests/dump_case_loader.cpp"
)

target_include_directories(dump_linear_systems
  PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${CUPF_ROOT}/cpp/inc"
    "${CUPF_ROOT}/cpp/src"
    "${CUPF_ROOT}/tests"
)

target_link_libraries(dump_linear_systems
  PRIVATE
    cupf
    iterative_linear_system_io
)

if(ITERATIVE_WITH_CUDA)
  set_target_properties(dump_linear_systems PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()

option(ITERATIVE_WITH_HYPRE "Build CPU HYPRE BoomerAMG experiment solver" ON)
set(ITERATIVE_PROBE_SOURCES
  bicgstab_block_jacobi.cpp
  bicgstab_diag.cpp
  bicgstab_identity.cpp
  bicgstab_ilu0.cpp
  bicgstab_ilu1.cpp
  bicgstab_ilut.cpp
  iterative_probe.cpp
  iterative_probe_common.cpp
  solver_dispatch.cpp
)

if(ITERATIVE_WITH_HYPRE)
  find_package(MPI REQUIRED)
  find_path(HYPRE_INCLUDE_DIR HYPRE.h PATH_SUFFIXES hypre)
  find_library(HYPRE_LIBRARY NAMES HYPRE)
  if(NOT HYPRE_INCLUDE_DIR OR NOT HYPRE_LIBRARY)
    message(FATAL_ERROR "ITERATIVE_WITH_HYPRE=ON but HYPRE was not found")
  endif()
  list(APPEND ITERATIVE_PROBE_SOURCES
    bicgstab_hypre_boomeramg.cpp
    fgmres_hypre_boomeramg.cpp
    hypre_boomeramg_common.cpp
  )
endif()

add_executable(iterative_probe
  ${ITERATIVE_PROBE_SOURCES}
)

target_include_directories(iterative_probe
  PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
)

target_link_libraries(iterative_probe
  PRIVATE
    iterative_linear_system_io
    Eigen3::Eigen
)

if(ITERATIVE_WITH_HYPRE)
  target_compile_definitions(iterative_probe PRIVATE ITERATIVE_WITH_HYPRE=1)
  target_include_directories(iterative_probe PRIVATE "${HYPRE_INCLUDE_DIR}")
  target_link_libraries(iterative_probe PRIVATE "${HYPRE_LIBRARY}" MPI::MPI_CXX)
endif()

set_target_properties(iterative_linear_system_io PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

set_target_properties(dump_linear_systems iterative_probe PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
