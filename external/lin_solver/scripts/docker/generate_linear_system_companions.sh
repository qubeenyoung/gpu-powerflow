#!/usr/bin/env bash
set -euo pipefail

tool="${LINEAR_SYSTEM_COMPANION_TOOL:-prepare_dataset_vectors}"
seed="${LINEAR_SYSTEM_RANDOM_SEED:-20260521}"
suite_root="${BENCHMARK_MATRIX_ROOT:-${SUITESPARSE_MATRIX_ROOT:-/datasets/benchmark_matrices}}/matrices"
nr_root="${POWER_NR_LINEAR_SYSTEM_ROOT:-${POWER_SYSTEM_DATASET_ROOT:-/datasets/power_system}/nr_linear_systems}"
manifest="${DATASETS_ROOT:-/datasets}/LINEAR_SYSTEM_COMPANIONS.txt"

command -v "${tool}" >/dev/null 2>&1 || {
    echo "Companion generation tool not found: ${tool}" >&2
    exit 1
}

mkdir -p "$(dirname "${manifest}")"

{
    printf 'Linear-system companion files\n'
    printf 'Tool: %s\n' "$(command -v "${tool}")"
    printf 'SuiteSparse seed: %s\n' "${seed}"
    printf 'Generated at: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${manifest}"

if [[ -d "${suite_root}" ]]; then
    while IFS= read -r -d '' matrix_dir; do
        matrix_name="$(basename "${matrix_dir}")"
        matrix_path="${matrix_dir}/${matrix_name}.mtx"
        if [[ ! -f "${matrix_path}" ]]; then
            matrix_path="$(
                find "${matrix_dir}" -maxdepth 1 -type f -name '*.mtx' \
                    ! -name '*_b.mtx' \
                    ! -name 'F.mtx' \
                    ! -name 'J.mtx' \
                    ! -name 'rhs.mtx' \
                    ! -name 'x_true.mtx' \
                    | sort \
                    | head -1
            )"
        fi
        if [[ -z "${matrix_path}" || ! -f "${matrix_path}" ]]; then
            printf '\n[skip] suitesparse %s: no matrix .mtx found\n' "${matrix_name}" >> "${manifest}"
            continue
        fi

        "${tool}" \
            --mode random-rhs \
            --matrix "${matrix_path}" \
            --rhs-out "${matrix_dir}/rhs.mtx" \
            --x-true-out "${matrix_dir}/x_true.mtx" \
            --seed "${seed}"

        {
            printf '\n[suitesparse] %s\n' "${matrix_name}"
            printf 'matrix: %s\n' "${matrix_path}"
            printf 'rhs: %s\n' "${matrix_dir}/rhs.mtx"
            printf 'x_true: %s\n' "${matrix_dir}/x_true.mtx"
        } >> "${manifest}"
    done < <(find "${suite_root}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
fi

if [[ -d "${nr_root}" ]]; then
    while IFS= read -r -d '' case_dir; do
        case_name="$(basename "${case_dir}")"
        matrix_path="${case_dir}/J.mtx"
        rhs_input_path="${case_dir}/F.mtx"
        if [[ ! -f "${matrix_path}" || ! -f "${rhs_input_path}" ]]; then
            printf '\n[skip] matpower_nr %s: missing J.mtx or F.mtx\n' "${case_name}" >> "${manifest}"
            continue
        fi

        "${tool}" \
            --mode solve-x \
            --matrix "${matrix_path}" \
            --rhs-in "${rhs_input_path}" \
            --rhs-out "${case_dir}/rhs.mtx" \
            --x-true-out "${case_dir}/x_true.mtx"

        {
            printf '\n[matpower_nr] %s\n' "${case_name}"
            printf 'matrix: %s\n' "${matrix_path}"
            printf 'rhs source: %s\n' "${rhs_input_path}"
            printf 'rhs: %s\n' "${case_dir}/rhs.mtx"
            printf 'x_true: %s\n' "${case_dir}/x_true.mtx"
        } >> "${manifest}"
    done < <(find "${nr_root}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
fi
