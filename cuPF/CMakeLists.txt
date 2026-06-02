cmake_minimum_required(VERSION 3.22)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(WITH_CUDA             "Build the CUDA backend"           OFF)
option(BUILD_PYTHON_BINDINGS "Build the pybind11 Python module" OFF)
option(CUPF_WITH_TORCH      "Build torch::Tensor zero-copy Python extension APIs" OFF)
option(BUILD_BENCHMARKS      "Build benchmark executables"      OFF)
option(BUILD_EVALUATORS      "Build cuPF evaluation executables" OFF)
option(ENABLE_LOG            "Enable spdlog logging when available" ON)
option(ENABLE_DUMP           "Enable debug dump utilities"      OFF)
option(ENABLE_TIMING         "Enable ScopedTimer timing collection" OFF)
option(ENABLE_NVTX           "Enable NVTX tracing utility when headers are available" OFF)
option(CUPF_ENABLE_CUSTOM_SOLVER "Build the custom CUDA FP64 linear solver backend" OFF)
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
find_package(spdlog QUIET)

if(CUPF_WITH_TORCH)
  if(NOT BUILD_PYTHON_BINDINGS)
    message(FATAL_ERROR "CUPF_WITH_TORCH=ON requires BUILD_PYTHON_BINDINGS=ON")
  endif()
  if(NOT WITH_CUDA)
    message(FATAL_ERROR "CUPF_WITH_TORCH=ON currently requires WITH_CUDA=ON")
  endif()
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE CUPF_TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(CUPF_TORCH_CMAKE_PREFIX)
    list(APPEND CMAKE_PREFIX_PATH "${CUPF_TORCH_CMAKE_PREFIX}")
  endif()
  find_package(Torch REQUIRED)
  find_library(TORCH_PYTHON_LIBRARY
    NAMES torch_python
    HINTS "${CUPF_TORCH_CMAKE_PREFIX}/../../lib" "${Python_SITEARCH}/torch/lib"
          /usr/local/lib/python3.10/dist-packages/torch/lib)
  if(NOT TORCH_PYTHON_LIBRARY)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')"
      OUTPUT_VARIABLE CUPF_TORCH_LIB_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    find_library(TORCH_PYTHON_LIBRARY NAMES torch_python HINTS "${CUPF_TORCH_LIB_DIR}")
  endif()
  if(NOT TORCH_PYTHON_LIBRARY)
    message(FATAL_ERROR "CUPF_WITH_TORCH=ON requires libtorch_python.so")
  endif()
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pathlib, sys; out=[]; [out.append(str(p)) for b in map(pathlib.Path, sys.path) for p in sorted((b / 'nvidia').glob('*/lib')) if p.is_dir() and str(p) not in out]; print(';'.join(out))"
    OUTPUT_VARIABLE CUPF_PYTHON_NVIDIA_LIBRARY_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pathlib, sys; libs=[]; [libs.extend(sorted((b / 'nvidia' / 'cublas' / 'lib').glob('libcublasLt.so*'))) for b in map(pathlib.Path, sys.path)]; print(libs[0] if libs else '')"
    OUTPUT_VARIABLE CUPF_PYTHON_CUBLASLT_LIBRARY
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  message(STATUS "Torch CMake prefix: ${CUPF_TORCH_CMAKE_PREFIX}")
endif()

set(CUPF_LOGGING_ENABLED OFF)
if(ENABLE_LOG AND spdlog_FOUND)
  set(CUPF_LOGGING_ENABLED ON)
elseif(ENABLE_LOG)
  message(STATUS "spdlog not found: disabling cuPF logging")
endif()

# SuiteSparse KLU (required for CPU backend)
find_path(SUITESPARSE_INCLUDE_DIR
  NAMES klu.h btf.h amd.h colamd.h SuiteSparse_config.h suitesparse_config.h
  PATHS /usr/include /usr/include/suitesparse
        /usr/local/include /usr/local/include/suitesparse)
find_library(LIB_KLU      NAMES klu      PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
find_library(LIB_UMFPACK  NAMES umfpack  PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
find_library(LIB_BTF      NAMES btf      PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
find_library(LIB_AMD      NAMES amd      PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
find_library(LIB_COLAMD   NAMES colamd   PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
find_library(LIB_SSCONFIG NAMES suitesparseconfig SuiteSparse_config
             PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)

if(NOT LIB_KLU OR NOT LIB_UMFPACK OR NOT LIB_BTF OR NOT LIB_AMD OR NOT LIB_COLAMD OR NOT LIB_SSCONFIG)
  message(FATAL_ERROR "cuPF requires SuiteSparse KLU/UMFPACK (klu, umfpack, btf, amd, colamd, suitesparseconfig).")
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

  # The Debian libcudss0-dev-cuda-12 package installs versioned subdirectories
  # (/usr/include/libcudss/12, /usr/lib/x86_64-linux-gnu/libcudss/12), which
  # differ from the flat /usr/include layout of the pip nvidia-cudss wheel, so
  # search both.
  find_path(CUDSS_INCLUDE_DIR
    NAMES cuDSS.h cudss.h
    HINTS
      /usr/include
      /usr/include/libcudss/12
      /usr/local/include
      /usr/local/cuda/include
      $ENV{HOME}/.local/lib/python3.10/site-packages/nvidia/cu12/include)
  find_library(CUDSS_LIBRARY
    NAMES cudss
    HINTS
      /usr/lib
      /usr/lib/x86_64-linux-gnu
      /usr/lib/x86_64-linux-gnu/libcudss/12
      /usr/local/lib
      $ENV{HOME}/.local/lib/python3.10/site-packages/nvidia/cu12/lib)

  if(NOT CUDSS_INCLUDE_DIR OR NOT CUDSS_LIBRARY)
    message(FATAL_ERROR "WITH_CUDA=ON requires cuDSS headers and library. Set CUDSS_INCLUDE_DIR/CUDSS_LIBRARY if needed.")
  endif()
endif()

if(CUPF_ENABLE_CUSTOM_SOLVER)
  if(NOT WITH_CUDA)
    message(FATAL_ERROR "CUPF_ENABLE_CUSTOM_SOLVER=ON requires WITH_CUDA=ON")
  endif()
  set(CLS_BUILD_CUDA_OPS ON CACHE BOOL "" FORCE)
  set(CLS_BUILD_SCRIPTS OFF CACHE BOOL "" FORCE)
  set(CLS_BUILD_CUDSS_SCRIPT OFF CACHE BOOL "" FORCE)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../custom_linear_solver"
                   "${CMAKE_CURRENT_BINARY_DIR}/custom_linear_solver")
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
  cpp/src/newton_solver/core/csr_transpose.cpp
  cpp/src/newton_solver/ops/jacobian/jacobian_analysis.cpp
  cpp/src/newton_solver/core/newton_solver.cpp
  cpp/src/newton_solver/core/newton_solver_adjoint.cpp
  cpp/src/newton_solver/core/adjoint_math.cpp
  cpp/src/newton_solver/core/torch_bridge.cpp

  # Storage — CPU
  cpp/src/newton_solver/storage/cpu/cpu_fp64_storage.cpp

  # Ops — CPU
  cpp/src/newton_solver/ops/ibus/compute_ibus.cpp
  cpp/src/newton_solver/ops/mismatch/cpu_mismatch.cpp
  cpp/src/newton_solver/ops/jacobian/fill_jacobian.cpp
  cpp/src/newton_solver/ops/linear_solve/cpu_klu.cpp
  cpp/src/newton_solver/ops/linear_solve/cpu_umfpack.cpp
  cpp/src/newton_solver/ops/voltage_update/cpu_voltage_update.cpp
)

# ---------------------------------------------------------------------------
# CUDA sources (batch-aware CUDA ops; single-case is B=1)
# ---------------------------------------------------------------------------
if(WITH_CUDA)
  set(CUPF_CUDA_SOURCES
    # Storage — CUDA (one templated impl for FP32/FP64/Mixed profiles)
    cpp/src/newton_solver/storage/cuda/cuda_batched_storage.cpp
    cpp/src/newton_solver/storage/cuda/storage_convert.cu

    # Ops — CUDA Ibus
    cpp/src/newton_solver/ops/ibus/compute_ibus.cu

    # Ops — CUDA mismatch
    cpp/src/newton_solver/ops/mismatch/cuda_mismatch.cu
    cpp/src/newton_solver/ops/mismatch/compute_mismatch_from_ibus.cu
    cpp/src/newton_solver/ops/mismatch/reduce_mismatch_norm.cu

    # Ops — CUDA jacobian
    cpp/src/newton_solver/ops/jacobian/fill_jacobian_gpu.cu

    # Ops — CUDA linear solve
    cpp/src/newton_solver/ops/linear_solve/linear_solve_kernels.cu
    cpp/src/newton_solver/ops/linear_solve/torch_bridge_kernels.cu
    cpp/src/newton_solver/ops/linear_solve/cuda_cudss.cpp

    # Ops — CUDA voltage update
    cpp/src/newton_solver/ops/voltage_update/cuda_voltage_update.cu
  )
  if(CUPF_ENABLE_CUSTOM_SOLVER)
    list(APPEND CUPF_CUDA_SOURCES
      cpp/src/newton_solver/ops/linear_solve/cuda_custom_solver.cpp
    )
  endif()
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
)

target_link_libraries(cupf PUBLIC
  ${LIB_KLU} ${LIB_UMFPACK} ${LIB_BTF} ${LIB_AMD} ${LIB_COLAMD} ${LIB_SSCONFIG}
  $<$<BOOL:${CUPF_LOGGING_ENABLED}>:spdlog::spdlog>
)

target_compile_definitions(cupf PRIVATE CUPF_INTERNAL_TORCH_API)

target_compile_definitions(cupf PUBLIC
  $<$<BOOL:${WITH_CUDA}>:CUPF_WITH_CUDA>
  $<$<BOOL:${WITH_CUDA}>:CUPF_ENABLE_CUDSS>
  $<$<BOOL:${CUPF_ENABLE_CUSTOM_SOLVER}>:CUPF_ENABLE_CUSTOM_SOLVER>
  $<$<BOOL:${WITH_CUDA}>:CUPF_CUDSS_REORDERING_ALG=${CUPF_CUDSS_REORDERING_ALG_DEFINE}>
  $<$<BOOL:${CUPF_CUDSS_ENABLE_MT}>:CUPF_CUDSS_ENABLE_MT>
  $<$<BOOL:${CUPF_CUDSS_SET_HOST_NTHREADS}>:CUPF_CUDSS_HOST_NTHREADS=${CUPF_CUDSS_HOST_NTHREADS_DEFINE}>
  $<$<BOOL:${CUPF_CUDSS_SET_ND_NLEVELS}>:CUPF_CUDSS_ND_NLEVELS=${CUPF_CUDSS_ND_NLEVELS_DEFINE}>
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
  if(CUPF_ENABLE_CUSTOM_SOLVER)
    target_link_libraries(cupf PUBLIC custom_linear_solver_api)
  endif()
  set_target_properties(cupf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# ---------------------------------------------------------------------------
# Python bindings
# ---------------------------------------------------------------------------
if(BUILD_PYTHON_BINDINGS)
  find_package(pybind11 REQUIRED)
  set(CUPF_PYBIND_SOURCES bindings/pybind_cupf.cpp)
  if(CUPF_WITH_TORCH)
    list(APPEND CUPF_PYBIND_SOURCES bindings/torch_cupf_extension.cpp)
  endif()
  pybind11_add_module(_cupf MODULE NO_EXTRAS ${CUPF_PYBIND_SOURCES})
  target_link_libraries(_cupf PRIVATE cupf)
  target_include_directories(_cupf PRIVATE cpp/inc cpp/src)
  if(CUPF_WITH_TORCH)
    get_filename_component(CUPF_TORCH_LIBRARY_DIR "${TORCH_PYTHON_LIBRARY}" DIRECTORY)
    target_link_libraries(_cupf PRIVATE
      $<$<BOOL:${CUPF_PYTHON_CUBLASLT_LIBRARY}>:-Wl,--no-as-needed>
      $<$<BOOL:${CUPF_PYTHON_CUBLASLT_LIBRARY}>:${CUPF_PYTHON_CUBLASLT_LIBRARY}>
      $<$<BOOL:${CUPF_PYTHON_CUBLASLT_LIBRARY}>:-Wl,--as-needed>
      ${TORCH_LIBRARIES}
      ${TORCH_PYTHON_LIBRARY})
    target_compile_definitions(_cupf PRIVATE CUPF_WITH_TORCH)
    target_compile_options(_cupf PRIVATE ${TORCH_CXX_FLAGS})
    target_include_directories(_cupf PRIVATE ${TORCH_INCLUDE_DIRS})
  endif()
  if(WITH_CUDA)
    get_filename_component(CUPF_CUDSS_LIBRARY_DIR "${CUDSS_LIBRARY}" DIRECTORY)
    set(CUPF_PYBIND_RPATH "${CUPF_CUDSS_LIBRARY_DIR}")
    if(CUPF_WITH_TORCH)
      foreach(_cupf_nvidia_library_dir IN LISTS CUPF_PYTHON_NVIDIA_LIBRARY_DIRS)
        string(APPEND CUPF_PYBIND_RPATH ";${_cupf_nvidia_library_dir}")
      endforeach()
      string(APPEND CUPF_PYBIND_RPATH ";${CUPF_TORCH_LIBRARY_DIR}")
    endif()
    string(APPEND CUPF_PYBIND_RPATH ";${CUDAToolkit_LIBRARY_DIR}")
    set_target_properties(_cupf PROPERTIES
      BUILD_RPATH "${CUPF_PYBIND_RPATH}"
      INSTALL_RPATH "${CUPF_PYBIND_RPATH}")
  endif()
  install(TARGETS _cupf LIBRARY DESTINATION cupf)
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
if(BUILD_BENCHMARKS)
  set(CUPF_BENCHMARKS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks" CACHE PATH
      "Path to cuPF benchmark sources")
  if(NOT EXISTS "${CUPF_BENCHMARKS_DIR}/CMakeLists.txt")
    get_filename_component(CUPF_WORKSPACE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
    set(CUPF_BENCHMARKS_DIR "${CUPF_WORKSPACE_ROOT}/benchmarks" CACHE PATH
        "Path to cuPF benchmark sources" FORCE)
  endif()
  if(NOT EXISTS "${CUPF_BENCHMARKS_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "BUILD_BENCHMARKS=ON but benchmark sources were not found. "
      "Set CUPF_BENCHMARKS_DIR to the benchmarks directory.")
  endif()
  add_subdirectory("${CUPF_BENCHMARKS_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/benchmarks")
endif()

# ---------------------------------------------------------------------------
# Tests and Evaluators
# ---------------------------------------------------------------------------
# The C++ evaluators (cupf_cpp_evaluate, cupf_batch_bench) live under tests/ next to the
# shared dump_case_loader they depend on, so tests/ is added whenever either testing or
# evaluators are requested.
if(BUILD_TESTING OR BUILD_EVALUATORS)
  add_subdirectory(tests)
endif()
