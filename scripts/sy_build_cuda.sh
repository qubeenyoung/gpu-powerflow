#!/usr/bin/env bash
# ============================================
# 소영CUDA: NewtonPF CUDA Build Script
# ============================================

# ---- [스크립트 위치 기준으로 프로젝트 루트 찾기] ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT" || { echo "에러: 프로젝트 루트를 찾을 수 없습니다."; exit 1; }

echo "프로젝트 루트: $PROJECT_ROOT"
echo

# ---- [프로젝트 옵션] ----
BUILD_DIR="build"
BUILD_TYPE="Release"
USE_AMGX="ON"  # 소영CUDA: Enable AmgX GPU solver
GENERATOR="Ninja"
NPROC="8"

# ---- [기능 토글] ----
BUILD_CPP=ON
BUILD_CUDA=ON          # 소영CUDA: CUDA 빌드 활성화
BUILD_AMGX=ON          # 소영CUDA: AmgX GPU sparse solver 활성화
BUILD_PYBIND=OFF
BUILD_TEST=ON

# ---- [매크로 토글] ----
ENABLE_BLOCK_TIMER=ON
ENABLE_DUMP_DATA=ON

# ---- [빌드 동작 제어] ----
DO_CLEAN=ON
VERBOSE=OFF
ONLY_CONFIGURE=OFF
EXTRA_CMAKE_ARGS=()

# ============================================
set -euo pipefail

# ---- 색상 출력 유틸 ----
c() {
  local color="$1"; shift
  local code=""
  case "$color" in
    red) code="31";; green) code="32";; yellow) code="33";;
    blue) code="34";; magenta) code="35";; cyan) code="36";;
  esac
  echo -e "\\033[${code}m$*\\033[0m"
}

# ---- 요약 ----
c cyan "== NewtonPF CUDA Build Summary =="
echo "BUILD_DIR          : $BUILD_DIR"
echo "BUILD_TYPE         : $BUILD_TYPE"
echo "GENERATOR          : $GENERATOR"
echo "NPROC              : $NPROC"
echo "BUILD_CPP          : $BUILD_CPP"
echo "BUILD_CUDA         : $BUILD_CUDA"  # 소영CUDA
echo "BUILD_AMGX         : $BUILD_AMGX"
echo "BUILD_PYBIND       : $BUILD_PYBIND"
echo "ENABLE_BLOCK_TIMER : $ENABLE_BLOCK_TIMER"
echo "ENABLE_DUMP_DATA   : $ENABLE_DUMP_DATA"
echo "DO_CLEAN           : $DO_CLEAN"
echo "VERBOSE            : $VERBOSE"
echo "ONLY_CONFIGURE     : $ONLY_CONFIGURE"
if ((${#EXTRA_CMAKE_ARGS[@]})); then
  echo "EXTRA_CMAKE_ARGS   : ${EXTRA_CMAKE_ARGS[*]}"
fi
echo

# ---- 클린 ----
if [[ "$DO_CLEAN" == "ON" ]]; then
  c yellow "[clean] Removing $BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

# ---- Configure ----
c blue "[configure] CMake configure → ${BUILD_DIR}"
cmake -S . -B "$BUILD_DIR" -G "$GENERATOR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DBUILD_CPP="$BUILD_CPP" \
  -DBUILD_CUDA="$BUILD_CUDA" \
  -DBUILD_AMGX="$BUILD_AMGX" \
  -DAMGX_LIBRARY="/usr/local/lib/libamgxsh.so" \
  -DAMGX_INCLUDE_DIR="/usr/local/include" \
  -DBUILD_PYBIND="$BUILD_PYBIND" \
  -DBUILD_TEST="$BUILD_TEST" \
  -DENABLE_BLOCK_TIMER="$ENABLE_BLOCK_TIMER" \
  -DENABLE_DUMP_DATA="$ENABLE_DUMP_DATA" \
  "${EXTRA_CMAKE_ARGS[@]}"

# ---- Build ----
if [[ "$ONLY_CONFIGURE" == "OFF" ]]; then
  c blue "[build] Building project (threads: $NPROC)"
  VFLAG=()
  [[ "$VERBOSE" == "ON" ]] && VFLAG+=("-v")
  cmake --build "$BUILD_DIR" -j"$NPROC" "${VFLAG[@]}"
else
  c yellow "[build] ONLY_CONFIGURE=ON → build 단계 생략"
fi

c green "CUDA Build completed."
