#!/usr/bin/env bash
set -u

EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${EXP_ROOT}/logs"
THIRD_PARTY="${EXP_ROOT}/third_party"
LOG_FILE="${LOG_DIR}/install_strumpack.log"
mkdir -p "${LOG_DIR}" "${THIRD_PARTY}"
exec > >(tee "${LOG_FILE}") 2>&1

echo "[strumpack] started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[strumpack] prefix: ${THIRD_PARTY}/strumpack/install"

if find /usr/local /usr "${THIRD_PARTY}" \( -name STRUMPACKConfig.cmake -o -name 'libstrumpack*' \) -print -quit 2>/dev/null | grep -q .; then
    echo "[strumpack] status=available"
    find /usr/local /usr "${THIRD_PARTY}" \( -name STRUMPACKConfig.cmake -o -name 'libstrumpack*' \) -print 2>/dev/null | head
    exit 0
fi

if ! command -v mpicc >/dev/null 2>&1 || ! command -v mpicxx >/dev/null 2>&1; then
    echo "[strumpack] status=build_failed"
    echo "[strumpack] failure=MPI C/C++ compilers not found on PATH"
    echo "[strumpack] missing_dependency=mpicc/mpicxx"
    echo "[strumpack] likely_system_command=sudo apt-get install -y libopenmpi-dev openmpi-bin"
    echo "[strumpack] attempted_command=preflight dependency check before git clone/build"
    exit 2
fi

for cmd in git cmake nvcc gfortran; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "[strumpack] status=build_failed"
        echo "[strumpack] missing_dependency=${cmd}"
        exit 2
    fi
done

SRC="${THIRD_PARTY}/strumpack/src"
BUILD="${THIRD_PARTY}/strumpack/build"
PREFIX="${THIRD_PARTY}/strumpack/install"

if [[ ! -d "${SRC}/.git" ]]; then
    echo "[strumpack] command=git clone --depth 1 https://github.com/pghysels/STRUMPACK.git ${SRC}"
    git clone --depth 1 https://github.com/pghysels/STRUMPACK.git "${SRC}" || {
        echo "[strumpack] status=build_failed"
        echo "[strumpack] failure=git clone failed"
        exit 3
    }
fi

mkdir -p "${BUILD}"
echo "[strumpack] command=cmake -S ${SRC} -B ${BUILD} -DSTRUMPACK_USE_CUDA=ON -DCMAKE_INSTALL_PREFIX=${PREFIX}"
cmake -S "${SRC}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DSTRUMPACK_USE_CUDA=ON \
    -DSTRUMPACK_USE_MPI=ON \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx || {
    echo "[strumpack] status=build_failed"
    echo "[strumpack] failure=cmake configure failed"
    exit 4
}

echo "[strumpack] command=cmake --build ${BUILD} -j$(nproc)"
cmake --build "${BUILD}" -j"$(nproc)" || {
    echo "[strumpack] status=build_failed"
    echo "[strumpack] failure=build failed"
    exit 5
}

echo "[strumpack] command=cmake --install ${BUILD}"
cmake --install "${BUILD}" || {
    echo "[strumpack] status=build_failed"
    echo "[strumpack] failure=install failed"
    exit 6
}

echo "[strumpack] status=installed"
