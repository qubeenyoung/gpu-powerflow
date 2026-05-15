cmake_minimum_required(VERSION 3.22)

project(superlu_dist_measurement_audit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP)

set(LIN_SOL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(COMMON_INCLUDE "${LIN_SOL_DIR}/common/cpp")
set(SUPERLU_DIST_PREFIX "${LIN_SOL_DIR}/third_party/superlu_dist/install")

find_library(SUPERLU_DIST_LIBRARY
    NAMES superlu_dist libsuperlu_dist.a
    PATHS "${SUPERLU_DIST_PREFIX}/lib"
    REQUIRED
)

find_path(SUPERLU_DIST_INCLUDE_DIR
    NAMES superlu_ddefs.h
    PATHS "${SUPERLU_DIST_PREFIX}/include"
    REQUIRED
)

function(add_superlu_audit_target target_name colperm_name rowperm_name)
    add_executable(${target_name} superlu_dist_audit_entry.cpp)
    target_include_directories(${target_name} PRIVATE
        "${COMMON_INCLUDE}"
        "${SUPERLU_DIST_INCLUDE_DIR}"
    )
    target_compile_definitions(${target_name} PRIVATE
        SUPERLU_AUDIT_COLPERM=${colperm_name}
        SUPERLU_AUDIT_ROWPERM=${rowperm_name}
        SUPERLU_DIST_LIBRARY_PATH_STR="${SUPERLU_DIST_LIBRARY}"
    )
    target_link_libraries(${target_name} PRIVATE
        MPI::MPI_CXX
        "${SUPERLU_DIST_LIBRARY}"
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
        CUDA::cusparse
        OpenMP::OpenMP_CXX
        /usr/lib/x86_64-linux-gnu/libblas.so
        /usr/lib/x86_64-linux-gnu/liblapack.so
        m
    )
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endfunction()

add_superlu_audit_target(superlu_dist_audit_natural NATURAL LargeDiag_MC64)
add_superlu_audit_target(superlu_dist_audit_mmd_at_plus_a MMD_AT_PLUS_A LargeDiag_MC64)
add_superlu_audit_target(superlu_dist_audit_mmd_ata MMD_ATA LargeDiag_MC64)
add_superlu_audit_target(superlu_dist_audit_natural_norowperm NATURAL NOROWPERM)
add_superlu_audit_target(superlu_dist_audit_mmd_at_plus_a_norowperm MMD_AT_PLUS_A NOROWPERM)
add_superlu_audit_target(superlu_dist_audit_mmd_ata_norowperm MMD_ATA NOROWPERM)
