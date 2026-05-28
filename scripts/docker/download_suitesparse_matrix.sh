#!/usr/bin/env bash
set -euo pipefail

matrix_root="${BENCHMARK_MATRIX_ROOT:-${SUITESPARSE_MATRIX_ROOT:-/datasets/benchmark_matrices}}"
matrix_urls="${SUITESPARSE_MATRIX_URLS:-${SUITESPARSE_MATRIX_URL:-}}"
download_dir="${matrix_root}/downloads"
extract_dir="${matrix_root}/matrices"
manifest="${matrix_root}/MANIFEST.txt"

mkdir -p "${download_dir}" "${extract_dir}"

matrix_urls="${matrix_urls//,/ }"
read -r -a matrix_url_list <<< "${matrix_urls}"

{
    printf 'SuiteSparse Matrix Collection benchmark downloads\n'
    printf 'Download root: %s\n' "${download_dir}"
    printf 'Extract root: %s\n' "${extract_dir}"
    printf 'Downloaded at: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${manifest}"

if [[ "${#matrix_url_list[@]}" -eq 0 || -z "${matrix_url_list[*]// }" ]]; then
    printf '\nNo SuiteSparse Matrix Collection URLs were configured.\n' >> "${manifest}"
    exit 0
fi

for index in "${!matrix_url_list[@]}"; do
    matrix_url="${matrix_url_list[$index]}"

    if [[ "${#matrix_url_list[@]}" -eq 1 ]]; then
        current_archive="$(basename "${matrix_url}")"
    else
        current_archive="$(printf '%04d_%s' "$((index + 1))" "$(basename "${matrix_url}")")"
    fi

    archive_path="${download_dir}/${current_archive}"
    curl -fL --retry 5 --retry-delay 2 "${matrix_url}" -o "${archive_path}"

    {
        printf '\nURL: %s\n' "${matrix_url}"
        printf 'Archive: %s\n' "${archive_path}"
    } >> "${manifest}"

    if tar -tf "${archive_path}" >/dev/null 2>&1; then
        tar -xf "${archive_path}" -C "${extract_dir}"
        {
            printf 'Extracted to: %s\n' "${extract_dir}"
            printf 'Contents:\n'
            tar -tf "${archive_path}"
        } >> "${manifest}"
    else
        printf 'Archive was not a tar-compatible file; kept as downloaded.\n' >> "${manifest}"
    fi
done
