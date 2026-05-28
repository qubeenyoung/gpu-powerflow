#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "ERROR: ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND}" >&2' ERR

ROOT="${THIRD_PARTY_ROOT:-/opt/third_party}"
SRC_ROOT="${THIRD_PARTY_SRC:-${ROOT}/src}"
BUILD_ROOT="${THIRD_PARTY_BUILD:-${ROOT}/build}"
INSTALL_ROOT="${THIRD_PARTY_INSTALL:-${ROOT}/install}"
COMMON_PREFIX="${THIRD_PARTY_COMMON_PREFIX:-${INSTALL_ROOT}/common}"
STRUMPACK_CPU_PREFIX="${THIRD_PARTY_STRUMPACK_CPU_PREFIX:-${INSTALL_ROOT}/strumpack-cpu}"
STRUMPACK_CUDA_PREFIX="${THIRD_PARTY_STRUMPACK_CUDA_PREFIX:-${INSTALL_ROOT}/strumpack-cuda}"
GLU3_PREFIX="${THIRD_PARTY_GLU3_PREFIX:-${INSTALL_ROOT}/glu3}"
PASTIX_CPU_PREFIX="${THIRD_PARTY_PASTIX_CPU_PREFIX:-${INSTALL_ROOT}/pastix-cpu}"
PASTIX_CUDA_PREFIX="${THIRD_PARTY_PASTIX_CUDA_PREFIX:-${INSTALL_ROOT}/pastix-cuda}"
PANGULU_PREFIX="${THIRD_PARTY_PANGULU_PREFIX:-${INSTALL_ROOT}/pangulu}"
MUMPS_PREFIX="${THIRD_PARTY_MUMPS_PREFIX:-${INSTALL_ROOT}/mumps}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-86}"
RELEASE_CFLAGS="${RELEASE_CFLAGS:--O3 -DNDEBUG}"
RELEASE_CXXFLAGS="${RELEASE_CXXFLAGS:--O3 -DNDEBUG}"
RELEASE_FFLAGS="${RELEASE_FFLAGS:--O3 -DNDEBUG}"
JOBS="${JOBS:-$(nproc)}"
MANIFEST="${ROOT}/VERSIONS.txt"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-/usr/local/cuda}"
export MKLROOT="${MKLROOT:-/opt/intel/oneapi/mkl/latest}"
export PATH="${CUDA_HOME}/bin:${PATH}"

mkdir -p "${SRC_ROOT}" "${BUILD_ROOT}" "${INSTALL_ROOT}" "${COMMON_PREFIX}" \
         "${STRUMPACK_CPU_PREFIX}" "${STRUMPACK_CUDA_PREFIX}" "${GLU3_PREFIX}" \
         "${PASTIX_CPU_PREFIX}" "${PASTIX_CUDA_PREFIX}" "${PANGULU_PREFIX}" \
         "${MUMPS_PREFIX}"

printf '# Built on %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${MANIFEST}"

record_repo() {
    printf '%-24s %s\n' "$1" "$2" >> "${MANIFEST}"
}

clone_repo() {
    local name="$1"
    local url="$2"
    local ref="$3"
    local recursive="${4:-0}"
    local dir="${SRC_ROOT}/${name}"

    if [[ -d "${dir}/.git" ]]; then
        return
    fi

    if [[ "${recursive}" == "1" ]]; then
        git clone --depth 1 --branch "${ref}" --recursive "${url}" "${dir}"
    else
        git clone --depth 1 --branch "${ref}" "${url}" "${dir}"
    fi

    record_repo "${name}" "${ref} (${url})"
}

log_step() {
    printf '\n==> %s\n' "$1"
}

find_mkl_lib_dir() {
    local candidate

    for candidate in \
        "${MKL_LIB_DIR:-}" \
        "${MKLROOT}/lib" \
        "${MKLROOT}/lib/intel64"; do
        if [[ -n "${candidate}" && -f "${candidate}/libmkl_core.so" ]]; then
            printf '%s\n' "${candidate}"
            return
        fi
    done

    echo "unable to locate oneMKL libraries under ${MKLROOT}" >&2
    exit 1
}

find_mkl_interface_lib() {
    local candidate

    for candidate in \
        "${MKL_LIB_DIR}/libmkl_intel_lp64.so" \
        "${MKL_LIB_DIR}/libmkl_gf_lp64.so"; do
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return
        fi
    done

    echo "unable to locate oneMKL LP64 interface library in ${MKL_LIB_DIR}" >&2
    exit 1
}

cmake_build() {
    local src_dir="$1"
    local build_dir="$2"
    local install_prefix="$3"
    shift 3

    cmake -S "${src_dir}" -B "${build_dir}" -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
        -DCMAKE_C_FLAGS_RELEASE="${RELEASE_CFLAGS}" \
        -DCMAKE_CXX_FLAGS_RELEASE="${RELEASE_CXXFLAGS}" \
        -DCMAKE_Fortran_FLAGS_RELEASE="${RELEASE_FFLAGS}" \
        "$@"

    cmake --build "${build_dir}" --parallel "${JOBS}"
    cmake --install "${build_dir}"
}

resolve_shared_library() {
    local name="$1"
    local resolved

    resolved="$(ldconfig -p | awk -v lib="${name}" '$1 == lib { print $NF; exit }')"
    if [[ -z "${resolved}" ]]; then
        echo "unable to resolve ${name} via ldconfig" >&2
        exit 1
    fi

    printf '%s\n' "${resolved}"
}

normalize_cmake_cuda_architectures() {
    local raw="${1:-86}"
    local token
    local -a tokens
    local normalized=""

    raw="${raw//,/;}"
    raw="${raw// /;}"
    while [[ "${raw}" == *";;"* ]]; do
        raw="${raw//;;/;}"
    done
    raw="${raw#;}"
    raw="${raw%;}"

    IFS=';' read -ra tokens <<< "${raw}"
    for token in "${tokens[@]}"; do
        [[ -n "${token}" ]] || continue
        token="${token#sm_}"
        token="${token#SM_}"
        if [[ ! "${token}" =~ ^[0-9]+[A-Za-z]?$ ]]; then
            echo "unsupported CUDA architecture '${token}'; use values like 80, 86, 89, 90, or sm_90" >&2
            exit 1
        fi
        normalized+="${normalized:+;}${token}"
    done

    printf '%s\n' "${normalized:-86}"
}

CUDA_ARCHITECTURES="$(normalize_cmake_cuda_architectures "${CUDA_ARCHITECTURES}")"
METIS_INCLUDE_DIR="${COMMON_PREFIX}/include"
METIS_LIB="${COMMON_PREFIX}/lib/libmetis.so"
PARMETIS_LIB="${COMMON_PREFIX}/lib/libparmetis.so"
OPENBLAS_LIB="$(resolve_shared_library libopenblas.so)"
LAPACK_LIB="$(resolve_shared_library liblapack.so)"
SCALAPACK_LIB="$(resolve_shared_library libscalapack-openmpi.so)"
MKL_INCLUDE_DIR="${MKL_INCLUDE_DIR:-${MKLROOT}/include}"
MKL_LIB_DIR="$(find_mkl_lib_dir)"
MKL_INTERFACE_LIB="$(find_mkl_interface_lib)"
MKL_SEQ_LIBS="${MKL_INTERFACE_LIB};${MKL_LIB_DIR}/libmkl_sequential.so;${MKL_LIB_DIR}/libmkl_core.so;-lm;-ldl;-lpthread"

export PKG_CONFIG_PATH="${COMMON_PREFIX}/lib/pkgconfig:${COMMON_PREFIX}/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="${COMMON_PREFIX}/lib:${COMMON_PREFIX}/lib64:${GLU3_PREFIX}/lib:${PANGULU_PREFIX}/lib:${MKL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${COMMON_PREFIX}:${GLU3_PREFIX}:${PANGULU_PREFIX}:${MKLROOT}:${CMAKE_PREFIX_PATH:-}"
export CPATH="${MKL_INCLUDE_DIR}:${CPATH:-}"

log_step "Build GKlib"
clone_repo gklib https://github.com/KarypisLab/GKlib.git "${GKLIB_REF:-master}"
cmake_build "${SRC_ROOT}/gklib" "${BUILD_ROOT}/gklib" "${COMMON_PREFIX}" \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_C_FLAGS_RELEASE="${RELEASE_CFLAGS} -D_POSIX_C_SOURCE=199309L"

log_step "Build METIS"
clone_repo metis https://github.com/KarypisLab/METIS.git "${METIS_REF:-master}" 1
pushd "${SRC_ROOT}/metis" >/dev/null
make distclean >/dev/null 2>&1 || true
python3 - <<'PY'
from pathlib import Path

# PaStiX validates METIS with a standalone METIS_NodeND link test.  METIS uses
# libm symbols internally, so keep that dependency on the shared lib itself.
path = Path("libmetis/CMakeLists.txt")
text = path.read_text()
text = text.replace("target_link_libraries(metis GKlib)", "target_link_libraries(metis GKlib m)")
path.write_text(text)
PY
make config shared=1 prefix="${COMMON_PREFIX}" gklib_path="${COMMON_PREFIX}"
make -j"${JOBS}"
make install
popd >/dev/null

log_step "Build ParMETIS"
clone_repo parmetis https://github.com/KarypisLab/ParMETIS.git "${PARMETIS_REF:-main}" 1
pushd "${SRC_ROOT}/parmetis" >/dev/null
make distclean >/dev/null 2>&1 || true
make config cc=mpicc shared=1 prefix="${COMMON_PREFIX}" gklib_path="${COMMON_PREFIX}" metis_path="${COMMON_PREFIX}"
make -j"${JOBS}"
make install
popd >/dev/null

if [[ ! -f "${METIS_LIB}" ]]; then
    METIS_LIB="$(find "${COMMON_PREFIX}" -name 'libmetis.so*' | head -n 1)"
fi
if [[ ! -f "${PARMETIS_LIB}" ]]; then
    PARMETIS_LIB="$(find "${COMMON_PREFIX}" -name 'libparmetis.so*' | head -n 1)"
fi

log_step "Build SuiteSparse"
clone_repo suitesparse https://github.com/DrTimothyAldenDavis/SuiteSparse.git "${SUITESPARSE_REF:-v7.12.1}"
cmake_build "${SRC_ROOT}/suitesparse" "${BUILD_ROOT}/suitesparse" "${COMMON_PREFIX}" \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_STATIC_LIBS=ON \
    -DBUILD_TESTING=OFF \
    -DBLA_VENDOR=Intel10_64lp_seq \
    -DBLAS_LIBRARIES="${MKL_SEQ_LIBS}" \
    -DBLAS_INCLUDE_DIRS="${MKL_INCLUDE_DIR}" \
    -DLAPACK_LIBRARIES="${MKL_SEQ_LIBS}" \
    -DLAPACK_INCLUDE_DIRS="${MKL_INCLUDE_DIR}" \
    -DSUITESPARSE_ENABLE_PROJECTS="amd;btf;camd;ccolamd;colamd;cholmod;cxsparse;ldl;klu;umfpack;spqr" \
    -DSUITESPARSE_DEMOS=OFF \
    -DSUITESPARSE_USE_STRICT=ON \
    -DSUITESPARSE_USE_CUDA=OFF \
    -DCHOLMOD_USE_CUDA=OFF \
    -DSPQR_USE_CUDA=OFF \
    -DSUITESPARSE_USE_PYTHON=OFF

log_step "Build SuperLU"
clone_repo superlu https://github.com/xiaoyeli/superlu.git "${SUPERLU_REF:-v7.0.0}"
cmake_build "${SRC_ROOT}/superlu" "${BUILD_ROOT}/superlu" "${COMMON_PREFIX}" \
    -DBUILD_SHARED_LIBS=ON \
    -DTPL_BLAS_LIBRARIES="${MKL_SEQ_LIBS}" \
    -DTPL_ENABLE_METISLIB=ON \
    -DTPL_METIS_INCLUDE_DIRS="${METIS_INCLUDE_DIR}" \
    -DTPL_METIS_LIBRARIES="${METIS_LIB}"

log_step "Build SuperLU_MT"
clone_repo superlu_mt https://github.com/xiaoyeli/superlu_mt.git "${SUPERLU_MT_REF:-v4.0.2}"
cmake_build "${SRC_ROOT}/superlu_mt" "${BUILD_ROOT}/superlu_mt" "${COMMON_PREFIX}" \
    -DBUILD_SHARED_LIBS=ON \
    -DPLAT=_OPENMP \
    -DTPL_BLAS_LIBRARIES="${MKL_SEQ_LIBS}" \
    -Denable_doc=OFF \
    -Denable_tests=OFF

log_step "Build GLU3.0"
clone_repo glu3 https://github.com/sheldonucr/GLU_public.git "${GLU3_REF:-master}"
pushd "${SRC_ROOT}/glu3" >/dev/null
make -C src distclean >/dev/null 2>&1 || true
GLU3_CUDA_ARCH="${CUDA_ARCHITECTURES%%;*}" python3 - <<'PY'
from pathlib import Path
import os

arch = os.environ["GLU3_CUDA_ARCH"]
path = Path("src/Makefile")
text = path.read_text()
text = text.replace(
    "NVCCFLAGS = -O3 -std=c++11 -Xcompiler $(LU_INC)",
    f"NVCCFLAGS = -O3 -std=c++11 $(LU_INC) -Xcompiler -fPIC "
    f"-gencode=arch=compute_{arch},code=sm_{arch} "
    f"-gencode=arch=compute_{arch},code=compute_{arch}",
)
path.write_text(text)
PY
# GLU's Makefile has incomplete dependency edges around NICSLU utility archives.
# Keep this target serial to avoid racing the final lu_cmd link against NICSLU.
make -C src -j1 lu_cmd
install -D -m 0755 src/lu_cmd "${GLU3_PREFIX}/bin/glu3-lu_cmd"
mkdir -p "${GLU3_PREFIX}/include" "${GLU3_PREFIX}/lib" "${GLU3_PREFIX}/share/glu3"
cp -a include/. "${GLU3_PREFIX}/include/"
cp -a docs README README.md LICENSE.txt "${GLU3_PREFIX}/share/glu3/"
find src/nicslu -type f \( -name '*.a' -o -name '*.so' \) -exec cp -a {} "${GLU3_PREFIX}/lib/" \;
popd >/dev/null

log_step "Build StarPU"
clone_repo starpu https://github.com/starpu-runtime/starpu.git "${STARPU_REF:-master}"
pushd "${SRC_ROOT}/starpu" >/dev/null
./autogen.sh
mkdir -p "${BUILD_ROOT}/starpu"
pushd "${BUILD_ROOT}/starpu" >/dev/null
"${SRC_ROOT}/starpu/configure" --prefix="${COMMON_PREFIX}" --enable-cuda --enable-shared --disable-static
make -j"${JOBS}"
make install
popd >/dev/null
popd >/dev/null

log_step "Build PaStiX CPU"
clone_repo pastix https://gitlab.inria.fr/solverstack/pastix.git "${PASTIX_REF:-master}" 1
cmake_build "${SRC_ROOT}/pastix" "${BUILD_ROOT}/pastix-cpu" "${PASTIX_CPU_PREFIX}" \
    -DBUILD_SHARED_LIBS=ON \
    -DBLA_VENDOR=Intel10_64lp_seq \
    -DPASTIX_INT64=OFF \
    -DPASTIX_ORDERING_SCOTCH=OFF \
    -DPASTIX_ORDERING_METIS=ON \
    -DPASTIX_WITH_MPI=OFF \
    -DPASTIX_WITH_STARPU=OFF \
    -DPASTIX_WITH_CUDA=OFF \
    -DMETIS_DIR="${COMMON_PREFIX}" \
    -DMETIS_INCDIR="${METIS_INCLUDE_DIR}" \
    -DMETIS_LIBDIR="${COMMON_PREFIX}/lib" \
    -DMETIS_INCLUDE_DIR="${METIS_INCLUDE_DIR}" \
    -DMETIS_INCLUDE_DIRS="${METIS_INCLUDE_DIR}" \
    -DMETIS_LIBRARY="${METIS_LIB}" \
    -DMETIS_LIBRARIES="${METIS_LIB}"

log_step "Build PaStiX CUDA"
cmake_build "${SRC_ROOT}/pastix" "${BUILD_ROOT}/pastix-cuda" "${PASTIX_CUDA_PREFIX}" \
    -DBUILD_SHARED_LIBS=ON \
    -DBLA_VENDOR=Intel10_64lp_seq \
    -DPASTIX_INT64=OFF \
    -DPASTIX_ORDERING_SCOTCH=OFF \
    -DPASTIX_ORDERING_METIS=ON \
    -DPASTIX_WITH_MPI=OFF \
    -DPASTIX_WITH_STARPU=ON \
    -DPASTIX_WITH_CUDA=ON \
    -DSTARPU_DIR="${COMMON_PREFIX}" \
    -DMETIS_DIR="${COMMON_PREFIX}" \
    -DMETIS_INCDIR="${METIS_INCLUDE_DIR}" \
    -DMETIS_LIBDIR="${COMMON_PREFIX}/lib" \
    -DMETIS_INCLUDE_DIR="${METIS_INCLUDE_DIR}" \
    -DMETIS_INCLUDE_DIRS="${METIS_INCLUDE_DIR}" \
    -DMETIS_LIBRARY="${METIS_LIB}" \
    -DMETIS_LIBRARIES="${METIS_LIB}" \
    -DCMAKE_C_FLAGS="-I${CUDA_HOME}/include" \
    -DCMAKE_CXX_FLAGS="-I${CUDA_HOME}/include" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}"

log_step "Build STRUMPACK CPU"
clone_repo strumpack https://github.com/pghysels/STRUMPACK.git "${STRUMPACK_REF:-master}" 1
export METIS_DIR="${COMMON_PREFIX}"
export PARMETIS_DIR="${COMMON_PREFIX}"
export SCALAPACK_DIR="$(dirname "${SCALAPACK_LIB}")"

cmake_build "${SRC_ROOT}/strumpack" "${BUILD_ROOT}/strumpack-cpu" "${STRUMPACK_CPU_PREFIX}" \
    -DBUILD_SHARED_LIBS=ON \
    -DSTRUMPACK_USE_MPI=ON \
    -DSTRUMPACK_USE_CUDA=OFF \
    -DTPL_ENABLE_PARMETIS=OFF \
    -DTPL_METIS_INCLUDE_DIRS="${METIS_INCLUDE_DIR}" \
    -DTPL_METIS_LIBRARIES="${METIS_LIB}" \
    -DTPL_BLAS_LIBRARIES="${OPENBLAS_LIB}" \
    -DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
    -DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}"

log_step "Build STRUMPACK CUDA"
cmake_build "${SRC_ROOT}/strumpack" "${BUILD_ROOT}/strumpack-cuda" "${STRUMPACK_CUDA_PREFIX}" \
    -DBUILD_SHARED_LIBS=ON \
    -DSTRUMPACK_USE_MPI=ON \
    -DSTRUMPACK_USE_CUDA=ON \
    -DTPL_ENABLE_PARMETIS=OFF \
    -DTPL_ENABLE_MAGMA=OFF \
    -DTPL_METIS_INCLUDE_DIRS="${METIS_INCLUDE_DIR}" \
    -DTPL_METIS_LIBRARIES="${METIS_LIB}" \
    -DTPL_BLAS_LIBRARIES="${OPENBLAS_LIB}" \
    -DTPL_LAPACK_LIBRARIES="${LAPACK_LIB}" \
    -DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}"

log_step "Build PanguLU"
clone_repo pangulu https://github.com/SuperScientificSoftwareLaboratory/PanguLU.git "${PANGULU_REF:-v5.0.0}"
pushd "${SRC_ROOT}/pangulu" >/dev/null
make clean >/dev/null 2>&1 || true
PANGULU_CUDA_ARCH="${CUDA_ARCHITECTURES%%;*}"
cat > make.inc <<EOF
COMPILE_LEVEL = -O3
CUDA_COMPILE_LEVEL =

CUDA_INC = -I${CUDA_HOME}/include
NVCC = nvcc \$(CUDA_COMPILE_LEVEL) \$(COMPILE_LEVEL)
NVCCFLAGS = \$(PANGULU_FLAGS) -w -Xptxas -dlcm=cg -gencode=arch=compute_${PANGULU_CUDA_ARCH},code=sm_${PANGULU_CUDA_ARCH} -gencode=arch=compute_${PANGULU_CUDA_ARCH},code=compute_${PANGULU_CUDA_ARCH} \$(CUDA_INC)

CC = gcc \$(COMPILE_LEVEL)
MPICC = mpicc \$(COMPILE_LEVEL)
OPENBLAS_INC = -I/usr/include
CFLAGS = \$(OPENBLAS_INC) \$(CUDA_INC) -fopenmp -lpthread -lm
MPICCFLAGS = \$(CFLAGS)
REORDERING_INC = -I../reordering_omp/include
METIS_INC = -I${METIS_INCLUDE_DIR}
PANGULU_EXTRA_LIBS = ${METIS_LIB} ${OPENBLAS_LIB} -L${CUDA_HOME}/lib64 -lcudart -lcusolver -lcublas -lstdc++
PANGULU_FLAGS = -DPANGULU_LOG_INFO -DCALCULATE_TYPE_R64 -DGPU_OPEN -DMETIS
EOF
python3 - <<'PY'
from pathlib import Path

lib_make = Path("lib/Makefile")
text = lib_make.read_text()
text = text.replace(
    "$(MPICC) $(MPICCFLAGS) -shared -fPIC -o $@ ./pangulu*.o",
    "$(MPICC) $(MPICCFLAGS) -shared -fPIC -o $@ ./pangulu*.o $(PANGULU_EXTRA_LIBS)",
)
lib_make.write_text(text)
PY
python3 - <<PY
from pathlib import Path

path = Path("examples/Makefile")
text = path.read_text()
text = text.replace("#LINK_METIS = /path/to/metis/lib/libmetis.so", "LINK_METIS = ${METIS_LIB}")
text = text.replace("LINK_OPENBLAS = /path/to/openblas/lib/libopenblas.so", "LINK_OPENBLAS = ${OPENBLAS_LIB}")
text = text.replace("LINK_CUDA = -L/path/to/cuda/lib64 -lcudart -lstdc++ -lcusolver -lcublas", "LINK_CUDA = -L${CUDA_HOME}/lib64 -lcudart -lstdc++ -lcusolver -lcublas")
path.write_text(text)
PY
make -j"${JOBS}"
install -D -m 0755 examples/pangulu_example.elf "${PANGULU_PREFIX}/bin/pangulu_example"
install -D -m 0644 lib/libpangulu.so "${PANGULU_PREFIX}/lib/libpangulu.so"
install -D -m 0644 lib/libpangulu.a "${PANGULU_PREFIX}/lib/libpangulu.a"
mkdir -p "${PANGULU_PREFIX}/include" "${PANGULU_PREFIX}/share/pangulu"
cp -a include/. "${PANGULU_PREFIX}/include/"
find reordering_omp/lib -type f \( -name '*.a' -o -name '*.so' \) -exec cp -a {} "${PANGULU_PREFIX}/lib/" \;
cp -a README.md LICENSE PanguLU_Users_Guide.pdf "${PANGULU_PREFIX}/share/pangulu/"
popd >/dev/null

log_step "Build MUMPS"
clone_repo mumps-superbuild https://github.com/scivision/mumps-superbuild.git "${MUMPS_SUPERBUILD_REF:-main}" 1
cmake_build "${SRC_ROOT}/mumps-superbuild" "${BUILD_ROOT}/mumps" "${MUMPS_PREFIX}" \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_SINGLE=ON \
    -DBUILD_DOUBLE=ON \
    -DBUILD_COMPLEX=ON \
    -DBUILD_COMPLEX16=ON \
    -Dparallel=ON \
    -Dopenmp=ON \
    -Dmetis=ON \
    -Dparmetis=ON \
    -Dscotch=ON \
    -DMUMPS_gpu=ON \
    -DCUDAToolkit_ROOT="${CUDA_HOME}" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_Fortran_COMPILER=mpifort

if ! grep -q '^MUMPS_gpu:BOOL=ON$' "${BUILD_ROOT}/mumps/CMakeCache.txt"; then
    echo "MUMPS GPU support was requested but CMake did not enable MUMPS_gpu" >&2
    exit 1
fi
