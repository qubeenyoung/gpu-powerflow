#!/usr/bin/env bash
set -u

EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${EXP_ROOT}/logs"
THIRD_PARTY="${EXP_ROOT}/third_party"
LOG_FILE="${LOG_DIR}/install_ginkgo.log"
mkdir -p "${LOG_DIR}" "${THIRD_PARTY}"
exec > >(tee "${LOG_FILE}") 2>&1

echo "[ginkgo] started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[ginkgo] prefix: ${THIRD_PARTY}/ginkgo/install"

if find /usr/local /usr "${THIRD_PARTY}" -name GinkgoConfig.cmake -print -quit 2>/dev/null | grep -q .; then
    echo "[ginkgo] status=available"
    find /usr/local /usr "${THIRD_PARTY}" -name GinkgoConfig.cmake -print 2>/dev/null | head
    exit 0
fi

if [[ "${1:-}" == "--probe-only" ]]; then
    echo "[ginkgo] status=unavailable"
    echo "[ginkgo] failure=no existing GinkgoConfig.cmake or libginkgo was found"
    echo "[ginkgo] build_command=${BASH_SOURCE[0]}"
    exit 2
fi

for cmd in git cmake nvcc g++; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "[ginkgo] status=build_failed"
        echo "[ginkgo] missing_dependency=${cmd}"
        exit 2
    fi
done

SRC="${THIRD_PARTY}/ginkgo/src"
BUILD="${THIRD_PARTY}/ginkgo/build"
PREFIX="${THIRD_PARTY}/ginkgo/install"

if [[ ! -d "${SRC}/.git" ]]; then
    echo "[ginkgo] command=git clone --depth 1 https://github.com/ginkgo-project/ginkgo.git ${SRC}"
    git clone --depth 1 https://github.com/ginkgo-project/ginkgo.git "${SRC}" || {
        echo "[ginkgo] status=build_failed"
        echo "[ginkgo] failure=git clone failed"
        exit 3
    }
fi

mkdir -p "${BUILD}"
echo "[ginkgo] command=cmake -S ${SRC} -B ${BUILD} -DGINKGO_BUILD_CUDA=ON -DGINKGO_BUILD_TESTS=OFF -DGINKGO_BUILD_BENCHMARKS=OFF -DCMAKE_INSTALL_PREFIX=${PREFIX}"
cmake -S "${SRC}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DGINKGO_BUILD_CUDA=ON \
    -DGINKGO_BUILD_OMP=ON \
    -DGINKGO_BUILD_REFERENCE=ON \
    -DGINKGO_BUILD_TESTS=OFF \
    -DGINKGO_BUILD_BENCHMARKS=OFF \
    -DGINKGO_BUILD_EXAMPLES=OFF || {
    echo "[ginkgo] status=build_failed"
    echo "[ginkgo] failure=cmake configure failed"
    exit 4
}

echo "[ginkgo] command=cmake --build ${BUILD} -j$(nproc)"
cmake --build "${BUILD}" -j"$(nproc)" || {
    echo "[ginkgo] status=build_failed"
    echo "[ginkgo] failure=build failed"
    exit 5
}

echo "[ginkgo] command=cmake --install ${BUILD}"
cmake --install "${BUILD}" || {
    echo "[ginkgo] status=build_failed"
    echo "[ginkgo] failure=install failed"
    exit 6
}

echo "[ginkgo] status=installed"
