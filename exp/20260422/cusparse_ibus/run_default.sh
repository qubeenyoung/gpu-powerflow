#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_DIR="${ROOT_DIR}/exp/20260422/cusparse_ibus"
BUILD_DIR="${EXP_DIR}/build"
RESULTS_DIR="${EXP_DIR}/results"

cmake -S "${EXP_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target cusparse_ibus_bench -j

mkdir -p "${RESULTS_DIR}"
"${BUILD_DIR}/cusparse_ibus_bench" \
  --dataset-root "${ROOT_DIR}/datasets/texas_univ_cases/cuPF_datasets" \
  --cases case_ACTIVSg200,case_ACTIVSg2000,Texas7k_20220923,case_ACTIVSg25k,case_ACTIVSg70k \
  --batches 1,4,8,16,64,256 \
  --warmup 3 \
  --repeats 20 \
  --output "${RESULTS_DIR}/ibus_cusparse_spmm.csv"
