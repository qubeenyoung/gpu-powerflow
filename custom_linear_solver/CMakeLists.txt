cmake_minimum_required(VERSION 3.22)

project(custom_linear_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(CLS_BUILD_CUDA_OPS "Build the CUDA multifrontal factor/solve kernels" ON)
# ON  (default): setup captures the factor/solve kernels into internal CUDA graphs that
#                factorize/solve replay (the standalone behavior).
# OFF: external/capturable mode. The factor/solve path issues its kernels directly onto a caller
#      stream (set_stream) with no internal graph and no host sync, so an OUTER stream capture
#      (e.g. cuPF's whole-iteration graph) can record them. cuPF forces this OFF when built with
#      CUPF_ENABLE_CUDA_GRAPH.
option(CLS_INTERNAL_GRAPH "Capture factor/solve into internal CUDA graphs" ON)
option(CLS_BUILD_SCRIPTS "Build custom_linear_solver command-line scripts" ON)

set(CLS_CUDA_ARCHITECTURES "86" CACHE STRING
  "CUDA architectures for custom_linear_solver_ops; must be >= 80 for TF32 mma")

foreach(_cls_arch IN LISTS CLS_CUDA_ARCHITECTURES)
  string(REGEX REPLACE "-.*$" "" _cls_arch_num "${_cls_arch}")
  if(_cls_arch_num MATCHES "^[0-9]+$" AND _cls_arch_num LESS 80)
    message(FATAL_ERROR
      "custom_linear_solver requires CLS_CUDA_ARCHITECTURES >= 80 for TF32 mma; "
      "got ${_cls_arch}.")
  endif()
endforeach()

add_library(custom_linear_solver_api INTERFACE)
target_include_directories(custom_linear_solver_api INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

if(CLS_BUILD_CUDA_OPS)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)

  find_path(METIS_INCLUDE_DIR
    NAMES metis.h
    PATHS /usr/include /usr/local/include /opt/third_party/install/common/include)
  find_library(METIS_LIBRARY
    NAMES metis
    PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib /opt/third_party/install/common/lib)

  if(NOT METIS_INCLUDE_DIR OR NOT METIS_LIBRARY)
    message(FATAL_ERROR "custom_linear_solver requires METIS for nested-dissection ordering.")
  endif()

  set(CLS_OP_SOURCES
    src/solver.cpp
    src/analyze/pattern/pattern_kernels.cu
    src/internal/plan/multifrontal_plan.cu
    src/analyze/plan/lower.cu
    src/analyze/pipeline.cpp
    src/internal/runtime/state.cu
    src/internal/runtime/setup.cu
    src/factorize/factorize.cu
    src/solve/solve.cu
    src/analyze/symbolic/elimination_tree.cpp
    src/analyze/symbolic/supernode.cpp
    src/analyze/symbolic/multifrontal.cpp
    src/analyze/reorder/metis_nd.cpp
  )

  add_library(custom_linear_solver_ops STATIC ${CLS_OP_SOURCES})
  target_include_directories(custom_linear_solver_ops PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${METIS_INCLUDE_DIR}
  )
  target_link_libraries(custom_linear_solver_ops PUBLIC
    CUDA::cudart
    CUDA::cublas
    ${METIS_LIBRARY}
  )
  if(CLS_INTERNAL_GRAPH)
    target_compile_definitions(custom_linear_solver_ops PRIVATE CLS_INTERNAL_GRAPH)
  endif()
  set_target_properties(custom_linear_solver_ops PROPERTIES
    CUDA_ARCHITECTURES "${CLS_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF
  )

  target_link_libraries(custom_linear_solver_api INTERFACE custom_linear_solver_ops)

  if(CLS_BUILD_SCRIPTS)
    add_executable(custom_linear_solver_run
      tests/runners/io.cpp
      tests/runners/run_custom_solver.cu
    )
    target_include_directories(custom_linear_solver_run PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runners
    )
    target_link_libraries(custom_linear_solver_run PRIVATE
      custom_linear_solver_api
    )
    set_target_properties(custom_linear_solver_run PROPERTIES
      CUDA_ARCHITECTURES "${CLS_CUDA_ARCHITECTURES}"
      CUDA_SEPARABLE_COMPILATION OFF
    )
  endif()
endif()
