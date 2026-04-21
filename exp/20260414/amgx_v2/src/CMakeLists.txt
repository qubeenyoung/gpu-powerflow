find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

get_filename_component(WORKSPACE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(CUPF_ROOT "${WORKSPACE_ROOT}/cuPF")

find_path(AMGX_INCLUDE_DIR amgx_c.h PATHS /usr/local/include)
find_library(AMGX_LIBRARY NAMES amgxsh amgx PATHS /usr/local/lib)
if(NOT AMGX_INCLUDE_DIR OR NOT AMGX_LIBRARY)
  message(FATAL_ERROR "AMGX was not found")
endif()

add_library(amgx_v2 STATIC
  assembly/bus_block_jacobian_assembler.cu
  assembly/bus_local_jacobian_assembler.cu
  assembly/bus_local_residual_assembler.cu
  assembly/bus_local_voltage_update.cu
  linear/amgx_preconditioner.cu
  linear/block_amgx_preconditioner.cu
  linear/bus_block_jacobi_preconditioner.cu
  linear/csr_spmv.cu
  linear/device_vector_ops.cu
  linear/fgmres_solver.cu
  model/bus_local_index.cpp
  model/bus_local_jacobian_pattern.cpp
  model/graph_ordering.cpp
  solver/jfnk_amgx_solver.cu
)

target_compile_features(amgx_v2 PUBLIC cxx_std_17)

target_include_directories(amgx_v2
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
  PRIVATE
    "${AMGX_INCLUDE_DIR}"
    "${CUPF_ROOT}/cpp/inc"
)

target_compile_definitions(amgx_v2 PUBLIC CUPF_WITH_CUDA)

target_link_libraries(amgx_v2
  PUBLIC
    CUDA::cudart
    CUDA::cublas
    Eigen3::Eigen
    "${AMGX_LIBRARY}"
    OpenMP::OpenMP_CXX
)

set_target_properties(amgx_v2 PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

add_executable(amgx_v2_probe
  tools/amgx_v2_probe.cpp
  "${CUPF_ROOT}/tests/dump_case_loader.cpp"
)

target_compile_features(amgx_v2_probe PRIVATE cxx_std_17)

target_include_directories(amgx_v2_probe
  PRIVATE
    "${CUPF_ROOT}/cpp/inc"
    "${CUPF_ROOT}/tests"
)

target_compile_definitions(amgx_v2_probe
  PRIVATE
    AMGX_V2_DEFAULT_DATASET_ROOT="${WORKSPACE_ROOT}/exp/20260414/amgx/cupf_dumps"
)

target_link_libraries(amgx_v2_probe
  PRIVATE
    amgx_v2
    CUDA::cudart
)

set_target_properties(amgx_v2_probe PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "/usr/local/lib"
)
