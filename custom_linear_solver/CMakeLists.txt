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
option(CLS_FUSE_TF32_TRAIL_EXTEND
  "Experimental: fuse TF32 tensor-core trailing drain with extend-add for big fronts (helps B=1, neutral/regresses at batch; see note 55)" OFF)
option(CLS_MID_TF32_TC
  "Experimental: use TF32 tensor-core trailing GEMM on eligible mid-tier fronts" OFF)
option(CLS_MID_TF32_TC_THREADS_128
  "Experimental: launch mid TF32 Tensor Core kernel with 128 threads instead of 256" OFF)
option(CLS_MID_TF32_DIRECT_SHARED
  "Experimental: use direct-shared TF32 Tensor Core trailing for mid fronts instead of blocked updates" OFF)
option(CLS_MID_TF32_LOW_TC
  "Experimental: allow TF32 Tensor Core direct-shared trailing on 33..48 mid-low fronts for medium cases" OFF)
option(CLS_MID_TF32_LOW_TC_FORCE_ALL
  "Diagnostic: allow TF32 Tensor Core direct-shared trailing on 33..48 mid-low fronts for all case sizes" OFF)
option(CLS_SMALL_FRONT_MAX_16
  "Experimental: classify only fsz<=16 as small so 17..32 fronts use the mid dispatch path" OFF)
option(CLS_MID_LOW_SPLIT
  "Experimental: split 33..48 and 49..64 mid-low fronts into separate dispatch buckets" OFF)
option(CLS_BIG_TF32_BLOCKED_TC
  "Experimental: use right-looking blocked TF32 Tensor Core update for eligible big fronts" OFF)
option(CLS_BIG_TF32_SHARED_THREADS_512
  "Experimental: launch shared-resident big-low TF32 Tensor Core kernel with 512 threads instead of 256" OFF)
option(CLS_TF32_COLUMN_USOLVE
  "Experimental: use column-owned U-panel triangular solve in direct-shared TF32 mid fronts" OFF)
option(CLS_TF32_OZAKI_TC2
  "Experimental: use two-component Ozaki-style TF32 head/tail Tensor Core products" OFF)
option(CLS_TF32_OZAKI_TC2_FIRST_ORDER
  "Experimental: omit the tail-tail product from CLS_TF32_OZAKI_TC2" OFF)
option(CLS_RESPECT_PANEL_CAP
  "Experimental: respect the requested panel cap instead of overriding it by matrix size" OFF)
option(CLS_BIG_LOW_SPLIT
  "Experimental: split big fronts at fsz<=159 into a separate dispatch bucket" OFF)
option(CLS_TC_CLOSURE_PANEL_AMALGAMATE
  "Diagnostic: merge consecutive panels only when closure-valid and Tensor-Core-routable" OFF)
set(CLS_MID_TF32_MIN_FSZ "48" CACHE STRING
  "Experimental: minimum fsz_cap routed to the mid TF32 Tensor Core kernel")
if(NOT CLS_MID_TF32_MIN_FSZ MATCHES "^[0-9]+$")
  message(FATAL_ERROR "CLS_MID_TF32_MIN_FSZ must be an integer")
endif()
set(CLS_SIBLING_PANEL_AMALGAMATE_CAP "8" CACHE STRING
  "Diagnostic: max ncols for sibling-panel amalgamation")
if(NOT CLS_SIBLING_PANEL_AMALGAMATE_CAP MATCHES "^[0-9]+$")
  message(FATAL_ERROR "CLS_SIBLING_PANEL_AMALGAMATE_CAP must be an integer")
endif()
set(CLS_PANEL_CHAIN_AMALGAMATE_CAP "32" CACHE STRING
  "Diagnostic: max ncols for panel-tree-chain amalgamation")
if(NOT CLS_PANEL_CHAIN_AMALGAMATE_CAP MATCHES "^[0-9]+$")
  message(FATAL_ERROR "CLS_PANEL_CHAIN_AMALGAMATE_CAP must be an integer")
endif()
set(CLS_VALIDATED_RUN_PANEL_AMALGAMATE_CAP "32" CACHE STRING
  "Diagnostic: max ncols for validated consecutive-run panel amalgamation")
if(NOT CLS_VALIDATED_RUN_PANEL_AMALGAMATE_CAP MATCHES "^[0-9]+$")
  message(FATAL_ERROR "CLS_VALIDATED_RUN_PANEL_AMALGAMATE_CAP must be an integer")
endif()
set(CLS_CLOSURE_PANEL_AMALGAMATE_CAP "32" CACHE STRING
  "Diagnostic: max ncols for closure-validated panel amalgamation")
if(NOT CLS_CLOSURE_PANEL_AMALGAMATE_CAP MATCHES "^[0-9]+$")
  message(FATAL_ERROR "CLS_CLOSURE_PANEL_AMALGAMATE_CAP must be an integer")
endif()
set(CLS_TC_CLOSURE_PANEL_AMALGAMATE_CAP "32" CACHE STRING
  "Diagnostic: max ncols for Tensor-Core-routable closure panel amalgamation")
if(NOT CLS_TC_CLOSURE_PANEL_AMALGAMATE_CAP MATCHES "^[0-9]+$")
  message(FATAL_ERROR "CLS_TC_CLOSURE_PANEL_AMALGAMATE_CAP must be an integer")
endif()
set(CLS_PAR_ND_DEPTH "4" CACHE STRING
  "Diagnostic: recursion depth for parallel nested dissection")
if(NOT CLS_PAR_ND_DEPTH MATCHES "^[0-9]+$")
  message(FATAL_ERROR "CLS_PAR_ND_DEPTH must be an integer")
endif()
set(CLS_PAR_ND_SMALL_BASE_THR "4000" CACHE STRING
  "Diagnostic: base METIS_NodeND threshold for parallel ND when n < 20000")
if(NOT CLS_PAR_ND_SMALL_BASE_THR MATCHES "^[0-9]+$")
  message(FATAL_ERROR "CLS_PAR_ND_SMALL_BASE_THR must be an integer")
endif()
set(CLS_PAR_ND_LARGE_BASE_THR "20000" CACHE STRING
  "Diagnostic: base METIS_NodeND threshold for parallel ND when n >= 20000")
if(NOT CLS_PAR_ND_LARGE_BASE_THR MATCHES "^[0-9]+$")
  message(FATAL_ERROR "CLS_PAR_ND_LARGE_BASE_THR must be an integer")
endif()
if(CLS_TF32_OZAKI_TC2_FIRST_ORDER)
  set(CLS_TF32_OZAKI_TC2 ON CACHE BOOL
      "Experimental: use two-component Ozaki-style TF32 head/tail Tensor Core products" FORCE)
endif()
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
  if(CLS_FUSE_TF32_TRAIL_EXTEND)
    target_compile_definitions(custom_linear_solver_ops PRIVATE CLS_FUSE_TF32_TRAIL_EXTEND)
  endif()
  if(CLS_MID_TF32_TC)
    target_compile_definitions(custom_linear_solver_ops PRIVATE CLS_MID_TF32_TC)
    target_compile_definitions(custom_linear_solver_ops PRIVATE
      CLS_MID_TF32_MIN_FSZ=${CLS_MID_TF32_MIN_FSZ})
  endif()
  if(CLS_MID_TF32_TC_THREADS_128)
    target_compile_definitions(custom_linear_solver_ops PRIVATE CLS_MID_TF32_TC_THREADS_128)
  endif()
  if(CLS_MID_TF32_DIRECT_SHARED)
    target_compile_definitions(custom_linear_solver_ops PRIVATE CLS_MID_TF32_DIRECT_SHARED)
  endif()
  if(CLS_MID_TF32_LOW_TC OR CLS_MID_TF32_LOW_TC_FORCE_ALL)
    target_compile_definitions(custom_linear_solver_ops PRIVATE CLS_MID_TF32_LOW_TC)
  endif()
  if(CLS_MID_TF32_LOW_TC_FORCE_ALL)
    target_compile_definitions(custom_linear_solver_ops PRIVATE CLS_MID_TF32_LOW_TC_FORCE_ALL)
  endif()
  if(CLS_SMALL_FRONT_MAX_16)
    target_compile_definitions(custom_linear_solver_ops PRIVATE CLS_SMALL_FRONT_MAX_16)
  endif()
  if(CLS_MID_LOW_SPLIT)
    target_compile_definitions(custom_linear_solver_ops PRIVATE CLS_MID_LOW_SPLIT)
  endif()
  if(CLS_BIG_TF32_BLOCKED_TC)
    target_compile_definitions(custom_linear_solver_ops PRIVATE CLS_BIG_TF32_BLOCKED_TC)
  endif()
  if(CLS_BIG_TF32_SHARED_THREADS_512)
    target_compile_definitions(custom_linear_solver_ops PRIVATE CLS_BIG_TF32_SHARED_THREADS_512)
  endif()
  if(CLS_TF32_COLUMN_USOLVE)
    target_compile_definitions(custom_linear_solver_ops PRIVATE CLS_TF32_COLUMN_USOLVE)
  endif()
  if(CLS_TF32_OZAKI_TC2)
    target_compile_definitions(custom_linear_solver_ops PRIVATE CLS_TF32_OZAKI_TC2)
  endif()
  if(CLS_TF32_OZAKI_TC2_FIRST_ORDER)
    target_compile_definitions(custom_linear_solver_ops PRIVATE CLS_TF32_OZAKI_TC2_FIRST_ORDER)
  endif()
  if(CLS_RESPECT_PANEL_CAP)
    target_compile_definitions(custom_linear_solver_ops PRIVATE CLS_RESPECT_PANEL_CAP)
  endif()
  if(CLS_BIG_LOW_SPLIT)
    target_compile_definitions(custom_linear_solver_ops PRIVATE CLS_BIG_LOW_SPLIT)
  endif()
  if(CLS_TC_CLOSURE_PANEL_AMALGAMATE)
    target_compile_definitions(custom_linear_solver_ops PRIVATE
      CLS_TC_CLOSURE_PANEL_AMALGAMATE
      CLS_TC_CLOSURE_PANEL_AMALGAMATE_CAP=${CLS_TC_CLOSURE_PANEL_AMALGAMATE_CAP})
  endif()
  target_compile_definitions(custom_linear_solver_ops PRIVATE
    CLS_PAR_ND_DEPTH=${CLS_PAR_ND_DEPTH}
    CLS_PAR_ND_SMALL_BASE_THR=${CLS_PAR_ND_SMALL_BASE_THR}
    CLS_PAR_ND_LARGE_BASE_THR=${CLS_PAR_ND_LARGE_BASE_THR})
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
