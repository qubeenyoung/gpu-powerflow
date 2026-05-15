#!/usr/bin/env bash
set -u

EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${EXP_ROOT}/logs"
THIRD_PARTY="${EXP_ROOT}/third_party"
LOG_FILE="${LOG_DIR}/install_superlu_dist.log"
mkdir -p "${LOG_DIR}" "${THIRD_PARTY}"
exec > >(tee "${LOG_FILE}") 2>&1

echo "[superlu_dist] started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[superlu_dist] prefix: ${THIRD_PARTY}/superlu_dist/install"

if find /usr/local /usr "${THIRD_PARTY}" \( -name SuperLU_DISTConfig.cmake -o -name 'libsuperlu_dist*' \) -print -quit 2>/dev/null | grep -q .; then
    echo "[superlu_dist] status=available"
    find /usr/local /usr "${THIRD_PARTY}" \( -name SuperLU_DISTConfig.cmake -o -name 'libsuperlu_dist*' \) -print 2>/dev/null | head
    exit 0
fi

if ! command -v mpicc >/dev/null 2>&1 || ! command -v mpicxx >/dev/null 2>&1; then
    echo "[superlu_dist] status=build_failed"
    echo "[superlu_dist] failure=MPI C/C++ compilers not found on PATH"
    echo "[superlu_dist] missing_dependency=mpicc/mpicxx"
    echo "[superlu_dist] likely_system_command=sudo apt-get install -y libopenmpi-dev openmpi-bin"
    echo "[superlu_dist] attempted_command=preflight dependency check before git clone/build"
    exit 2
fi

for cmd in git cmake nvcc; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "[superlu_dist] status=build_failed"
        echo "[superlu_dist] missing_dependency=${cmd}"
        exit 2
    fi
done

SRC="${THIRD_PARTY}/superlu_dist/src"
BUILD="${THIRD_PARTY}/superlu_dist/build"
PREFIX="${THIRD_PARTY}/superlu_dist/install"

if [[ ! -d "${SRC}/.git" ]]; then
    echo "[superlu_dist] command=git clone --depth 1 https://github.com/xiaoyeli/superlu_dist.git ${SRC}"
    git clone --depth 1 https://github.com/xiaoyeli/superlu_dist.git "${SRC}" || {
        echo "[superlu_dist] status=build_failed"
        echo "[superlu_dist] failure=git clone failed"
        exit 3
    }
fi

mkdir -p "${BUILD}"
echo "[superlu_dist] command=cmake -S ${SRC} -B ${BUILD} -DTPL_ENABLE_CUDA=ON -DCMAKE_INSTALL_PREFIX=${PREFIX}"
cmake -S "${SRC}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DTPL_ENABLE_CUDA=ON \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx || {
    echo "[superlu_dist] status=build_failed"
    echo "[superlu_dist] failure=cmake configure failed"
    exit 4
}

echo "[superlu_dist] command=cmake --build ${BUILD} -j$(nproc)"
cmake --build "${BUILD}" -j"$(nproc)" || {
    echo "[superlu_dist] status=build_failed"
    echo "[superlu_dist] failure=build failed"
    exit 5
}

echo "[superlu_dist] command=cmake --install ${BUILD}"
cmake --install "${BUILD}" || {
    echo "[superlu_dist] status=build_failed"
    echo "[superlu_dist] failure=install failed"
    exit 6
}

echo "[superlu_dist] status=installed"
