cmake_minimum_required(VERSION 3.22)

project(custom_linear_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(CLS_BUILD_CUDA_OPS "Build copied CUDA multifrontal operation kernels" ON)
# ON  (default): batched_setup captures the factor/solve kernels into internal CUDA graphs that
#                batched_factorize/solve replay (the standalone behavior).
# OFF: external/capturable mode. The batched path issues its kernels directly onto a caller stream
#      (batched_set_stream) with no internal graph and no host sync, so an OUTER stream capture
#      (e.g. cuPF's whole-iteration graph) can record them. cuPF forces this OFF when built with
#      CUPF_ENABLE_CUDA_GRAPH.
option(CLS_INTERNAL_GRAPH "Capture batched factor/solve into internal CUDA graphs" ON)
option(CLS_BUILD_SCRIPTS "Build custom_linear_solver command-line scripts" ON)
option(CLS_BUILD_CUDSS_SCRIPT "Build cuDSS command-line comparison script" OFF)
option(CLS_ENABLE_NVTX "Enable NVTX ranges in command-line profiling scripts" OFF)
option(CLS_FUSE_FP16_TRAIL_EXTEND
  "Fuse FP16 tensor-core trailing drain with extend-add for big fronts" ON)
option(CLS_TF32X3
  "Experimental: error-corrected (Ootomo) TF32 tensor-core trailing for FP32-accurate results on big fronts" OFF)

# Compile-time gate for the profile/dump toolkit (src/profile/). Default ON for Debug builds,
# OFF for Release/RelWithDebInfo/MinSizeRel — when OFF, all profile macros expand to no-ops,
# profile.cpp is not compiled, and libnvToolsExt is not linked.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(_CLS_PROFILE_DEFAULT ON)
else()
  set(_CLS_PROFILE_DEFAULT OFF)
endif()
option(CLS_ENABLE_PROFILE
  "Compile in profile/dump instrumentation (CSV timers + buffer dumps)" ${_CLS_PROFILE_DEFAULT})

set(CLS_CUDA_ARCHITECTURES "86" CACHE STRING
  "CUDA architectures for custom_linear_solver_ops; must be >= 80 for TF32 WMMA")

foreach(_cls_arch IN LISTS CLS_CUDA_ARCHITECTURES)
  string(REGEX REPLACE "-.*$" "" _cls_arch_num "${_cls_arch}")
  if(_cls_arch_num MATCHES "^[0-9]+$" AND _cls_arch_num LESS 80)
    message(FATAL_ERROR
      "custom_linear_solver requires CLS_CUDA_ARCHITECTURES >= 80 for TF32 WMMA; "
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
    src/matrix/pattern_kernels.cu
    src/plan/multifrontal_plan.cu
    src/plan/analyze.cu
    src/plan/build.cpp
    src/numeric_engine.cu
    src/symbolic/elimination_tree.cpp
    src/symbolic/supernode.cpp
    src/symbolic/multifrontal.cpp
    src/reordering/metis_nd.cpp
  )

  # Profile module: compiled in only when CLS_ENABLE_PROFILE=ON. Pulls in NVTX as a link
  # dependency since profile.cpp uses libnvToolsExt directly. When OFF, profile.hpp inlines
  # empty stubs and none of this is needed.
  if(CLS_ENABLE_PROFILE)
    list(APPEND CLS_OP_SOURCES src/profile/profile.cpp)
    find_path(NVTX_INCLUDE_DIR
      NAMES nvtx3/nvToolsExt.h nvToolsExt.h
      HINTS
        /usr/local/cuda/include
        /usr/local/cuda-12.8/targets/x86_64-linux/include
        $ENV{HOME}/.local/lib/python3.10/site-packages/nvidia/nvtx/include
        /usr/local/lib/python3.10/dist-packages/nvidia/nvtx/include)
    find_library(NVTOOLSEXT_LIBRARY
      NAMES nvToolsExt
      HINTS
        /usr/local/cuda/lib64
        /usr/local/cuda-12.8/targets/x86_64-linux/lib
        /usr/lib
        /usr/lib/x86_64-linux-gnu)
    if(NOT NVTX_INCLUDE_DIR OR NOT NVTOOLSEXT_LIBRARY)
      message(FATAL_ERROR
        "CLS_ENABLE_PROFILE=ON requires NVTX headers and libnvToolsExt.")
    endif()
    get_filename_component(NVTOOLSEXT_LIBRARY_DIR "${NVTOOLSEXT_LIBRARY}" DIRECTORY)
  endif()

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
  if(CLS_ENABLE_PROFILE)
    target_compile_definitions(custom_linear_solver_ops PUBLIC CLS_ENABLE_PROFILE=1)
    target_include_directories(custom_linear_solver_ops PUBLIC ${NVTX_INCLUDE_DIR})
    target_link_libraries(custom_linear_solver_ops PUBLIC ${NVTOOLSEXT_LIBRARY})
  endif()
  if(CLS_INTERNAL_GRAPH)
    target_compile_definitions(custom_linear_solver_ops PRIVATE CLS_INTERNAL_GRAPH)
  endif()
  if(CLS_FUSE_FP16_TRAIL_EXTEND)
    target_compile_definitions(custom_linear_solver_ops PRIVATE CLS_FUSE_FP16_TRAIL_EXTEND)
  endif()
  if(CLS_TF32X3)
    target_compile_definitions(custom_linear_solver_ops PRIVATE CLS_TF32X3)
  endif()
  set_target_properties(custom_linear_solver_ops PROPERTIES
    CUDA_ARCHITECTURES "${CLS_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF
  )

  target_link_libraries(custom_linear_solver_api INTERFACE custom_linear_solver_ops)

  if(CLS_BUILD_SCRIPTS)
    # When CLS_ENABLE_NVTX is set, both runners get NVTX include + library so their
    # NVTX_RANGE_* helpers compile through to libnvToolsExt. Located here so the
    # detection runs once before either target is configured.
    if(CLS_ENABLE_NVTX)
      find_path(NVTX_INCLUDE_DIR
        NAMES nvtx3/nvToolsExt.h nvToolsExt.h
        HINTS
          /usr/local/cuda/include
          /usr/local/cuda-12.8/targets/x86_64-linux/include
          $ENV{HOME}/.local/lib/python3.10/site-packages/nvidia/nvtx/include
          /usr/local/lib/python3.10/dist-packages/nvidia/nvtx/include)
      find_library(NVTOOLSEXT_LIBRARY
        NAMES nvToolsExt
        HINTS
          /usr/local/cuda/lib64
          /usr/local/cuda-12.8/targets/x86_64-linux/lib
          /usr/lib
          /usr/lib/x86_64-linux-gnu)
      if(NOT NVTX_INCLUDE_DIR OR NOT NVTOOLSEXT_LIBRARY)
        message(FATAL_ERROR "CLS_ENABLE_NVTX=ON requires NVTX headers and libnvToolsExt.")
      endif()
      get_filename_component(NVTOOLSEXT_LIBRARY_DIR "${NVTOOLSEXT_LIBRARY}" DIRECTORY)
    endif()

    add_executable(custom_linear_solver_run
      tests/io.cpp
      tests/run_custom_solver.cu
    )
    target_include_directories(custom_linear_solver_run PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(custom_linear_solver_run PRIVATE
      custom_linear_solver_api
    )
    set_target_properties(custom_linear_solver_run PROPERTIES
      CUDA_ARCHITECTURES "${CLS_CUDA_ARCHITECTURES}"
      CUDA_SEPARABLE_COMPILATION OFF
    )
    if(CLS_ENABLE_NVTX)
      target_compile_definitions(custom_linear_solver_run PRIVATE CLS_ENABLE_NVTX=1)
      target_include_directories(custom_linear_solver_run PRIVATE ${NVTX_INCLUDE_DIR})
      target_link_libraries(custom_linear_solver_run PRIVATE ${NVTOOLSEXT_LIBRARY})
      set_target_properties(custom_linear_solver_run PROPERTIES
        BUILD_RPATH "${NVTOOLSEXT_LIBRARY_DIR}")
    endif()

    if(CLS_BUILD_CUDSS_SCRIPT)
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
        message(FATAL_ERROR "CLS_BUILD_CUDSS_SCRIPT=ON requires cuDSS headers/library.")
      endif()
      get_filename_component(CUDSS_LIBRARY_DIR "${CUDSS_LIBRARY}" DIRECTORY)

      add_executable(cudss_run
        tests/io.cpp
        tests/run_cudss_solver.cpp
      )
      target_include_directories(cudss_run PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CUDSS_INCLUDE_DIR}
      )
      target_link_libraries(cudss_run PRIVATE
        CUDA::cudart
        ${CUDSS_LIBRARY}
      )
      set(CUDSS_RUN_RPATH "${CUDSS_LIBRARY_DIR}")
      if(CLS_ENABLE_NVTX)
        # NVTX include / library are detected once above (next to the
        # custom_linear_solver_run target). Reuse the cached vars here.
        target_compile_definitions(cudss_run PRIVATE CLS_ENABLE_NVTX=1)
        target_include_directories(cudss_run PRIVATE ${NVTX_INCLUDE_DIR})
        target_link_libraries(cudss_run PRIVATE ${NVTOOLSEXT_LIBRARY})
        set(CUDSS_RUN_RPATH "${CUDSS_RUN_RPATH};${NVTOOLSEXT_LIBRARY_DIR}")
      endif()
      set_target_properties(cudss_run PROPERTIES
        BUILD_RPATH "${CUDSS_RUN_RPATH}"
      )
    endif()
  endif()
endif()
