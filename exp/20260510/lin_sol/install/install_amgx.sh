#!/usr/bin/env bash
set -u

EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${EXP_ROOT}/logs"
THIRD_PARTY="${EXP_ROOT}/third_party"
LOG_FILE="${LOG_DIR}/install_amgx.log"
mkdir -p "${LOG_DIR}" "${THIRD_PARTY}"
exec > >(tee "${LOG_FILE}") 2>&1

echo "[amgx] started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[amgx] prefix: ${THIRD_PARTY}/amgx/install"

if [[ -f /usr/local/include/amgx_c.h && ( -f /usr/local/lib/libamgxsh.so || -f /usr/local/lib/libamgx.a ) ]]; then
    echo "[amgx] status=available"
    echo "[amgx] include=/usr/local/include/amgx_c.h"
    ls -l /usr/local/lib/libamgx* || true
    exit 0
fi

for cmd in git cmake nvcc; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "[amgx] status=build_failed"
        echo "[amgx] missing_dependency=${cmd}"
        echo "[amgx] install_hint=Install ${cmd} without sudo from this script, or provide it on PATH."
        exit 2
    fi
done

SRC="${THIRD_PARTY}/amgx/src"
BUILD="${THIRD_PARTY}/amgx/build"
PREFIX="${THIRD_PARTY}/amgx/install"

if [[ ! -d "${SRC}/.git" ]]; then
    echo "[amgx] command=git clone --depth 1 https://github.com/NVIDIA/AMGX.git ${SRC}"
    git clone --depth 1 https://github.com/NVIDIA/AMGX.git "${SRC}" || {
        echo "[amgx] status=build_failed"
        echo "[amgx] failure=git clone failed"
        exit 3
    }
fi

mkdir -p "${BUILD}"
echo "[amgx] command=cmake -S ${SRC} -B ${BUILD} -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PREFIX}"
cmake -S "${SRC}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" || {
    echo "[amgx] status=build_failed"
    echo "[amgx] failure=cmake configure failed"
    exit 4
}

echo "[amgx] command=cmake --build ${BUILD} -j$(nproc)"
cmake --build "${BUILD}" -j"$(nproc)" || {
    echo "[amgx] status=build_failed"
    echo "[amgx] failure=build failed"
    exit 5
}

echo "[amgx] command=cmake --install ${BUILD}"
cmake --install "${BUILD}" || {
    echo "[amgx] status=build_failed"
    echo "[amgx] failure=install failed"
    exit 6
}

echo "[amgx] status=installed"
