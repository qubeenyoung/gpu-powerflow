cmake_minimum_required(VERSION 3.22)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(WITH_CUDA             "Build the CUDA backend"           OFF)
option(BUILD_PYTHON_BINDINGS "Build the pybind11 Python module" OFF)
option(BUILD_BENCHMARKS      "Build benchmark executables"      OFF)
option(ENABLE_LOG            "Enable spdlog logging when available" ON)
option(ENABLE_DUMP           "Enable debug dump utilities"      OFF)
option(ENABLE_TIMING         "Enable ScopedTimer timing collection" OFF)
option(ENABLE_NVTX           "Enable NVTX tracing utility when headers are available" OFF)
set(CUPF_CUDSS_REORDERING_ALG "DEFAULT" CACHE STRING "cuDSS reordering algorithm: DEFAULT, ALG_1, or ALG_2")
set_property(CACHE CUPF_CUDSS_REORDERING_ALG PROPERTY STRINGS DEFAULT ALG_1 ALG_2)
option(CUPF_CUDSS_ENABLE_MT "Enable cuDSS multi-threaded mode via CUDSS_THREADING_LIB" OFF)
set(CUPF_CUDSS_HOST_NTHREADS "AUTO" CACHE STRING "cuDSS host thread count: AUTO or an integer >= 1")
set(CUPF_CUDSS_ND_NLEVELS "AUTO" CACHE STRING "cuDSS nested dissection level count: AUTO or an integer >= 0")

project(cuPF LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
find_package(spdlog QUIET)

set(CUPF_LOGGING_ENABLED OFF)
if(ENABLE_LOG AND spdlog_FOUND)
  set(CUPF_LOGGING_ENABLED ON)
elseif(ENABLE_LOG)
  message(STATUS "spdlog not found: disabling cuPF logging")
endif()

set(CUPF_SUPERLU_ENABLED OFF)
find_path(SUPERLU_INCLUDE_DIR
  NAMES slu_ddefs.h
  PATHS /usr/include/superlu /usr/local/include/superlu)
find_library(LIB_SUPERLU
  NAMES superlu
  PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
if(SUPERLU_INCLUDE_DIR AND LIB_SUPERLU)
  set(CUPF_SUPERLU_ENABLED ON)
else()
  message(STATUS "SuperLU not found: disabling CpuSuperLU kernel variant")
endif()

# SuiteSparse KLU (required for CPU backend)
find_path(SUITESPARSE_INCLUDE_DIR
  NAMES klu.h btf.h amd.h colamd.h SuiteSparse_config.h suitesparse_config.h
  PATHS /usr/include /usr/include/suitesparse
        /usr/local/include /usr/local/include/suitesparse)
find_library(LIB_KLU      NAMES klu      PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
find_library(LIB_BTF      NAMES btf      PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
find_library(LIB_AMD      NAMES amd      PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
find_library(LIB_COLAMD   NAMES colamd   PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
find_library(LIB_SSCONFIG NAMES suitesparseconfig SuiteSparse_config
             PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)

if(NOT LIB_KLU OR NOT LIB_BTF OR NOT LIB_AMD OR NOT LIB_COLAMD OR NOT LIB_SSCONFIG)
  message(FATAL_ERROR "cuPF requires SuiteSparse KLU (klu, btf, amd, colamd, suitesparseconfig).")
endif()

# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------
set(CUPF_CUDSS_HOST_NTHREADS_DEFINE 0)
set(CUPF_CUDSS_SET_HOST_NTHREADS OFF)
set(CUPF_CUDSS_ND_NLEVELS_DEFINE 0)
set(CUPF_CUDSS_SET_ND_NLEVELS OFF)

if(WITH_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)

  string(TOUPPER "${CUPF_CUDSS_REORDERING_ALG}" CUPF_CUDSS_REORDERING_ALG_NORMALIZED)
  if(CUPF_CUDSS_REORDERING_ALG_NORMALIZED STREQUAL "DEFAULT")
    set(CUPF_CUDSS_REORDERING_ALG_DEFINE CUDSS_ALG_DEFAULT)
  elseif(CUPF_CUDSS_REORDERING_ALG_NORMALIZED STREQUAL "ALG_1")
    set(CUPF_CUDSS_REORDERING_ALG_DEFINE CUDSS_ALG_1)
  elseif(CUPF_CUDSS_REORDERING_ALG_NORMALIZED STREQUAL "ALG_2")
    set(CUPF_CUDSS_REORDERING_ALG_DEFINE CUDSS_ALG_2)
  else()
    message(FATAL_ERROR "CUPF_CUDSS_REORDERING_ALG must be one of DEFAULT, ALG_1, ALG_2; got '${CUPF_CUDSS_REORDERING_ALG}'")
  endif()
  message(STATUS "cuDSS reordering algorithm: ${CUPF_CUDSS_REORDERING_ALG_NORMALIZED}")

  string(TOUPPER "${CUPF_CUDSS_HOST_NTHREADS}" CUPF_CUDSS_HOST_NTHREADS_NORMALIZED)
  if(CUPF_CUDSS_HOST_NTHREADS_NORMALIZED STREQUAL "AUTO")
    set(CUPF_CUDSS_HOST_NTHREADS_DEFINE 0)
  elseif(CUPF_CUDSS_HOST_NTHREADS MATCHES "^[0-9]+$")
    if(CUPF_CUDSS_HOST_NTHREADS LESS 1)
      message(FATAL_ERROR "CUPF_CUDSS_HOST_NTHREADS must be AUTO or an integer >= 1; got '${CUPF_CUDSS_HOST_NTHREADS}'")
    endif()
    set(CUPF_CUDSS_SET_HOST_NTHREADS ON)
    set(CUPF_CUDSS_HOST_NTHREADS_DEFINE ${CUPF_CUDSS_HOST_NTHREADS})
  else()
    message(FATAL_ERROR "CUPF_CUDSS_HOST_NTHREADS must be AUTO or an integer >= 1; got '${CUPF_CUDSS_HOST_NTHREADS}'")
  endif()
  message(STATUS "cuDSS MT mode: ${CUPF_CUDSS_ENABLE_MT}")
  message(STATUS "cuDSS host threads: ${CUPF_CUDSS_HOST_NTHREADS}")

  string(TOUPPER "${CUPF_CUDSS_ND_NLEVELS}" CUPF_CUDSS_ND_NLEVELS_NORMALIZED)
  if(CUPF_CUDSS_ND_NLEVELS_NORMALIZED STREQUAL "AUTO")
    set(CUPF_CUDSS_ND_NLEVELS_DEFINE 0)
  elseif(CUPF_CUDSS_ND_NLEVELS MATCHES "^[0-9]+$")
    if(NOT CUPF_CUDSS_REORDERING_ALG_NORMALIZED STREQUAL "DEFAULT")
      message(FATAL_ERROR "CUPF_CUDSS_ND_NLEVELS only works with CUPF_CUDSS_REORDERING_ALG=DEFAULT")
    endif()
    set(CUPF_CUDSS_SET_ND_NLEVELS ON)
    set(CUPF_CUDSS_ND_NLEVELS_DEFINE ${CUPF_CUDSS_ND_NLEVELS})
  else()
    message(FATAL_ERROR "CUPF_CUDSS_ND_NLEVELS must be AUTO or an integer >= 0; got '${CUPF_CUDSS_ND_NLEVELS}'")
  endif()
  message(STATUS "cuDSS ND_NLEVELS: ${CUPF_CUDSS_ND_NLEVELS}")

  find_path(CUDSS_INCLUDE_DIR
    NAMES cuDSS.h cudss.h
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
    message(FATAL_ERROR "WITH_CUDA=ON requires cuDSS headers and library. Set CUDSS_INCLUDE_DIR/CUDSS_LIBRARY if needed.")
  endif()
endif()

set(CUPF_NVTX_ENABLED OFF)
if(ENABLE_NVTX)
  set(_cupf_nvtx_search_paths
    /usr/local/cuda/include
    /usr/local/cuda-12/include
    /usr/local/cuda-12.2/include)
  if(WITH_CUDA)
    list(APPEND _cupf_nvtx_search_paths ${CUDAToolkit_INCLUDE_DIRS})
  endif()

  find_path(NVTX_INCLUDE_DIR
    NAMES nvtx3/nvToolsExt.h nvToolsExt.h
    PATHS ${_cupf_nvtx_search_paths})

  if(NVTX_INCLUDE_DIR)
    set(CUPF_NVTX_ENABLED ON)
  else()
    message(STATUS "NVTX headers not found: disabling cuPF NVTX tracing")
  endif()
endif()

# ---------------------------------------------------------------------------
# Core sources
# ---------------------------------------------------------------------------
set(CUPF_CORE_SOURCES
  # Core
  cpp/src/newton_solver/ops/jacobian/jacobian_analysis.cpp
  cpp/src/newton_solver/core/newton_solver.cpp

  # Storage — CPU
  cpp/src/newton_solver/storage/cpu/cpu_fp64_storage.cpp

  # Ops — CPU
  cpp/src/newton_solver/ops/ibus/compute_ibus.cpp
  cpp/src/newton_solver/ops/mismatch/cpu_mismatch.cpp
  cpp/src/newton_solver/ops/jacobian/fill_jacobian.cpp
  cpp/src/newton_solver/ops/linear_solve/cpu_klu.cpp
  cpp/src/newton_solver/ops/voltage_update/cpu_voltage_update.cpp
)

# ---------------------------------------------------------------------------
# CUDA sources (batch-aware CUDA ops; single-case is B=1)
# ---------------------------------------------------------------------------
if(WITH_CUDA)
  set(CUPF_CUDA_SOURCES
    # Storage — CUDA
    cpp/src/newton_solver/storage/cuda/cuda_mixed_storage.cpp
    cpp/src/newton_solver/storage/cuda/cuda_fp64_storage.cpp

    # Ops — CUDA Ibus
    cpp/src/newton_solver/ops/ibus/compute_ibus.cu

    # Ops — CUDA mismatch
    cpp/src/newton_solver/ops/mismatch/cuda_mismatch.cu
    cpp/src/newton_solver/ops/mismatch/compute_mismatch_from_ibus.cu
    cpp/src/newton_solver/ops/mismatch/reduce_mismatch_norm.cu

    # Ops — CUDA jacobian
    cpp/src/newton_solver/ops/jacobian/fill_jacobian_gpu.cu

    # Ops — CUDA linear solve
    cpp/src/newton_solver/ops/linear_solve/prepare_rhs.cu
    cpp/src/newton_solver/ops/linear_solve/cuda_cudss.cpp

    # Ops — CUDA voltage update
    cpp/src/newton_solver/ops/voltage_update/cuda_voltage_update.cu
  )
endif()

# ---------------------------------------------------------------------------
# Library target
# ---------------------------------------------------------------------------
add_library(cupf STATIC ${CUPF_CORE_SOURCES} $<$<BOOL:${WITH_CUDA}>:${CUPF_CUDA_SOURCES}>)

target_include_directories(cupf PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/inc>
  ${SUITESPARSE_INCLUDE_DIR}
)

if(CUPF_NVTX_ENABLED)
  target_include_directories(cupf PUBLIC ${NVTX_INCLUDE_DIR})
endif()

if(WITH_CUDA)
  target_include_directories(cupf PUBLIC ${CUDSS_INCLUDE_DIR})
endif()

target_include_directories(cupf PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src
  $<$<BOOL:${CUPF_SUPERLU_ENABLED}>:${SUPERLU_INCLUDE_DIR}>
)

target_link_libraries(cupf PUBLIC
  Eigen3::Eigen
  ${LIB_KLU} ${LIB_BTF} ${LIB_AMD} ${LIB_COLAMD} ${LIB_SSCONFIG}
  $<$<BOOL:${CUPF_SUPERLU_ENABLED}>:${LIB_SUPERLU}>
  $<$<BOOL:${CUPF_LOGGING_ENABLED}>:spdlog::spdlog>
)

target_compile_definitions(cupf PUBLIC
  $<$<BOOL:${WITH_CUDA}>:CUPF_WITH_CUDA>
  $<$<BOOL:${WITH_CUDA}>:CUPF_ENABLE_CUDSS>
  $<$<BOOL:${WITH_CUDA}>:CUPF_CUDSS_REORDERING_ALG=${CUPF_CUDSS_REORDERING_ALG_DEFINE}>
  $<$<BOOL:${CUPF_CUDSS_ENABLE_MT}>:CUPF_CUDSS_ENABLE_MT>
  $<$<BOOL:${CUPF_CUDSS_SET_HOST_NTHREADS}>:CUPF_CUDSS_HOST_NTHREADS=${CUPF_CUDSS_HOST_NTHREADS_DEFINE}>
  $<$<BOOL:${CUPF_CUDSS_SET_ND_NLEVELS}>:CUPF_CUDSS_ND_NLEVELS=${CUPF_CUDSS_ND_NLEVELS_DEFINE}>
  $<$<BOOL:${CUPF_SUPERLU_ENABLED}>:CUPF_WITH_SUPERLU>
  $<$<BOOL:${CUPF_LOGGING_ENABLED}>:CUPF_ENABLE_LOG>
  $<$<BOOL:${ENABLE_DUMP}>:CUPF_ENABLE_DUMP>
  $<$<BOOL:${ENABLE_TIMING}>:CUPF_ENABLE_TIMING>
  $<$<BOOL:${CUPF_NVTX_ENABLED}>:CUPF_ENABLE_NVTX>
)

if(WITH_CUDA)
  target_link_libraries(cupf PUBLIC
    CUDA::cudart
    CUDA::cusparse
    ${CUDSS_LIBRARY}
  )
  set_target_properties(cupf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# ---------------------------------------------------------------------------
# Python bindings
# ---------------------------------------------------------------------------
if(BUILD_PYTHON_BINDINGS)
  find_package(pybind11 REQUIRED)
  pybind11_add_module(_cupf bindings/pybind_cupf.cpp)
  target_link_libraries(_cupf PRIVATE cupf)
  target_include_directories(_cupf PRIVATE cpp/inc)
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
