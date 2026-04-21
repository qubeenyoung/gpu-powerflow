find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(CUPF_ROOT "${REPO_ROOT}/cuPF")

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
find_library(LIB_METIS    NAMES metis libmetis.so.5
             PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)

if(NOT SUITESPARSE_INCLUDE_DIR OR NOT LIB_KLU OR NOT LIB_BTF OR
   NOT LIB_AMD OR NOT LIB_COLAMD OR NOT LIB_SSCONFIG)
  message(FATAL_ERROR "block_ilu_probe --compare-direction requires SuiteSparse KLU.")
endif()
if(NOT LIB_METIS)
  message(FATAL_ERROR "METIS library is required for --j11-partition metis.")
endif()

add_library(block_ilu_pf STATIC
  assembly/reduced_jacobian_assembler.cu
  linear/bicgstab_solver.cu
  linear/block_ilu_preconditioner.cu
  linear/csr_spmv.cu
  linear/cusparse_ilu0_block.cu
  linear/cusparse_ilu0_block_f32.cu
  linear/exact_klu_j11_f32.cpp
  linear/gmres_solver.cu
  linear/implicit_schur_operator.cu
  linear/implicit_schur_operator_f32.cu
  linear/metis_partition.cpp
  linear/partitioned_dense_lu_j11.cu
  linear/partitioned_dense_lu_j11_f32.cu
  linear/permuted_ilu0_block.cu
  linear/schur_bicgstab_solver.cu
  linear/schur_bicgstab_solver_f32.cu
  linear/schur_gmres_solver_f32.cu
  model/reduced_jacobian.cpp
  solver/powerflow_block_ilu_solver.cu
)

target_compile_features(block_ilu_pf PUBLIC cxx_std_17)

target_include_directories(block_ilu_pf
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CUPF_ROOT}/cpp/inc"
    "${SUITESPARSE_INCLUDE_DIR}"
)

target_compile_definitions(block_ilu_pf PUBLIC CUPF_WITH_CUDA)

target_link_libraries(block_ilu_pf
  PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    CUDA::cusparse
    Eigen3::Eigen
    ${LIB_KLU} ${LIB_BTF} ${LIB_AMD} ${LIB_COLAMD} ${LIB_SSCONFIG}
    ${LIB_METIS}
)

set_target_properties(block_ilu_pf PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

add_executable(block_ilu_probe
  tools/block_ilu_probe.cpp
  "${CUPF_ROOT}/tests/dump_case_loader.cpp"
)

target_compile_features(block_ilu_probe PRIVATE cxx_std_17)

target_include_directories(block_ilu_probe
  PRIVATE
    "${CUPF_ROOT}/tests"
    "${SUITESPARSE_INCLUDE_DIR}"
)

target_compile_definitions(block_ilu_probe
  PRIVATE
    BLOCK_ILU_DEFAULT_DATASET_ROOT="${REPO_ROOT}/exp/20260414/amgx/cupf_dumps"
)

target_link_libraries(block_ilu_probe
  PRIVATE
    block_ilu_pf
    Eigen3::Eigen
    ${LIB_KLU} ${LIB_BTF} ${LIB_AMD} ${LIB_COLAMD} ${LIB_SSCONFIG}
)

set_target_properties(block_ilu_probe PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
