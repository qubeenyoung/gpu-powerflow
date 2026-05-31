#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# experiment.sh — the cuPF "remove Eigen from the CPU backend" evaluator.
#
# This is the PROJECT's own evaluator: it lives in the project repo
# (cuPF/.agent/eigen-klu/) and travels/versions with the project. The common bot
# does not know anything about Eigen — it just runs this script in an isolated
# `docker run --rm` sibling and reads the metrics this script emits.
#
# Runs INSIDE the isolated experiment container. Configures + builds the cuPF CPU
# backend (no CUDA), runs the dependency-free smoke test cupf_minimal_tests, and
# counts the remaining Eigen usage. Emits machine-readable metrics to
# RESULTS_DIR/experiment.json. Always exits 0 so the orchestrator can read metrics
# even on a failed build (intermediate states are expected while Eigen is removed).
#
# Args: SRC_DIR BUILD_DIR RESULTS_DIR
# ---------------------------------------------------------------------------
set -uo pipefail

SRC="${1:?usage: experiment.sh SRC BUILD RESULTS}"
BUILD="${2:?missing BUILD dir}"
RESULTS="${3:?missing RESULTS dir}"

mkdir -p "$BUILD" "$RESULTS"
BUILD_LOG="$RESULTS/build.log"
TEST_LOG="$RESULTS/test.log"
: > "$BUILD_LOG"
: > "$TEST_LOG"

configure_ok=false
build_ok=false
test_ok=false
tests_passed=0
tests_total=0
build_seconds=0

# --- project metric: Eigen still present in the CPU-compiled files ----------
# These are the CPU (WITH_CUDA=OFF) translation units; reference/ files are not
# compiled and are out of scope. Paths are relative to SRC.
CPU_EIGEN_FILES=(
  "cpp/src/newton_solver/storage/cpu/cpu_fp64_storage.hpp"
  "cpp/src/newton_solver/storage/cpu/cpu_fp64_storage.cpp"
  "cpp/src/newton_solver/ops/linear_solve/cpu_klu.hpp"
  "cpp/src/newton_solver/ops/linear_solve/cpu_klu.cpp"
  "cpp/src/newton_solver/ops/ibus/compute_ibus.cpp"
  "cpp/src/newton_solver/ops/mismatch/cpu_mismatch.cpp"
  "cpp/src/newton_solver/ops/jacobian/fill_jacobian.cpp"
  "cpp/src/newton_solver/core/newton_solver_adjoint.cpp"
)
cpu_eigen_ref_files=0
for rel in "${CPU_EIGEN_FILES[@]}"; do
    f="$SRC/$rel"
    if [ -f "$f" ] && grep -Eq '#[[:space:]]*include[[:space:]]*<Eigen/' "$f"; then
        cpu_eigen_ref_files=$((cpu_eigen_ref_files + 1))
    fi
done
cmake_eigen_refs=0
if [ -f "$SRC/CMakeLists.txt" ]; then
    cmake_eigen_refs=$(grep -Eo 'Eigen3' "$SRC/CMakeLists.txt" | wc -l | tr -d ' ')
fi

echo "[experiment] cmake configure (WITH_CUDA=OFF, BUILD_TESTING=ON)" | tee -a "$BUILD_LOG"
if cmake -S "$SRC" -B "$BUILD" -G Ninja \
        -DWITH_CUDA=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release \
        >>"$BUILD_LOG" 2>&1; then
    configure_ok=true
fi

if [ "$configure_ok" = true ]; then
    echo "[experiment] cmake build --target cupf_minimal_tests" | tee -a "$BUILD_LOG"
    t0=$(date +%s)
    if cmake --build "$BUILD" --target cupf_minimal_tests >>"$BUILD_LOG" 2>&1; then
        build_ok=true
    fi
    t1=$(date +%s)
    build_seconds=$(( t1 - t0 ))
fi

if [ "$build_ok" = true ]; then
    echo "[experiment] ctest -R cupf_minimal_tests" | tee -a "$TEST_LOG"
    if ctest --test-dir "$BUILD" -R cupf_minimal_tests --output-on-failure >>"$TEST_LOG" 2>&1; then
        test_ok=true
    fi
    tests_total=$(grep -Eo 'out of [0-9]+' "$TEST_LOG" | tail -1 | grep -Eo '[0-9]+' || echo 0)
    if [ "$test_ok" = true ]; then tests_passed="$tests_total"; fi
fi

python3 - "$RESULTS/experiment.json" \
    "$configure_ok" "$build_ok" "$test_ok" \
    "$tests_passed" "$tests_total" "$build_seconds" \
    "$cpu_eigen_ref_files" "$cmake_eigen_refs" <<'PY'
import json, sys
out, cfg, bld, tst, tp, tt, secs, eig, ceig = sys.argv[1:10]
data = {
    "configure_ok": cfg == "true",
    "build_ok": bld == "true",
    "test_ok": tst == "true",
    "tests_passed": int(tp),
    "tests_total": int(tt),
    "build_seconds": int(secs),
    "cpu_eigen_ref_files": int(eig),
    "cmake_eigen_refs": int(ceig),
}
with open(out, "w") as fh:
    json.dump(data, fh, indent=2)
    fh.write("\n")
print("[experiment] wrote", out, data)
PY

exit 0
