#!/usr/bin/env bash
set -euo pipefail

power_dataset_root="${POWER_SYSTEM_DATASET_ROOT:-/datasets/power_system}"
matpower_dataset_root="${MATPOWER_DATASET_ROOT:-${power_dataset_root}/matpower}"
matpower_mat_root="${MATPOWER_MAT_ROOT:-${power_dataset_root}/matpower_mat}"
nr_output_root="${POWER_NR_LINEAR_SYSTEM_ROOT:-${power_dataset_root}/nr_linear_systems}"
convert_script_root="${POWER_CONVERT_SCRIPT_ROOT:-/opt/prepare-datasets/python}"
dump_iteration="${POWER_NR_DUMP_ITERATION:-2}"
cases="${POWER_NR_CASES:-case30 case118 case1197 case_ACTIVSg2000 case3012wp case6468rte case8387pegase case_ACTIVSg25k case_SyntheticUSA}"

cases="${cases//,/ }"
read -r -a case_list <<< "${cases}"

if [[ "${#case_list[@]}" -eq 0 ]]; then
    echo "POWER_NR_CASES is empty; skipping Newton-Raphson dataset generation"
    exit 0
fi

for script_name in convert_m2mat.py prepare_nr_linear_system.py common.py; do
    if [[ ! -f "${convert_script_root}/${script_name}" ]]; then
        echo "Missing conversion script: ${convert_script_root}/${script_name}" >&2
        exit 1
    fi
done

if [[ ! -d "${matpower_dataset_root}" ]]; then
    echo "MATPOWER dataset root not found: ${matpower_dataset_root}" >&2
    exit 1
fi

rm -rf "${matpower_mat_root}" "${nr_output_root}"
mkdir -p "${matpower_mat_root}" "${nr_output_root}"

python3 "${convert_script_root}/convert_m2mat.py" \
    --input-root "${matpower_dataset_root}" \
    --output-root "${matpower_mat_root}" \
    --cases "${case_list[@]}"

python3 "${convert_script_root}/prepare_nr_linear_system.py" \
    --mat-root "${matpower_mat_root}" \
    --output-root "${nr_output_root}" \
    --cases "${case_list[@]}" \
    --dump-iteration "${dump_iteration}"

{
    printf 'Newton-Raphson power-system linear systems\n'
    printf 'MATPOWER case root: %s\n' "${matpower_dataset_root}"
    printf 'MATPOWER .mat root: %s\n' "${matpower_mat_root}"
    printf 'Output root: %s\n' "${nr_output_root}"
    printf 'Dump iteration: %s\n' "${dump_iteration}"
    printf 'Generated at: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'Cases:\n'
    for case_name in "${case_list[@]}"; do
        printf '%s\n' "${case_name}"
    done
    printf 'Generated systems:\n'
    find "${nr_output_root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
} > "${nr_output_root}/MANIFEST.txt"
