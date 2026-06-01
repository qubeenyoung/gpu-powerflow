#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

if [[ -f "${repo_root}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${repo_root}/.env"
  set +a
fi

matlab_bin="${MATLAB_BIN:-matlab}"
case_name="${1:-${MATPOWER_CASE:-case9}}"
output_path="${2:-${MATPOWER_RESULT_JSON:-}}"

if [[ -n "${MATLAB_LICENSE_FILE:-}" && -z "${MLM_LICENSE_FILE:-}" ]]; then
  export MLM_LICENSE_FILE="${MATLAB_LICENSE_FILE}"
fi

export MATPOWER_HOME="${MATPOWER_HOME:-/opt/matpower}"

if [[ -z "${MATLAB_LICMODE:-}" && -n "${MATLAB_USER_ID:-}" && -n "${MATLAB_PASSWORD:-}" ]]; then
  MATLAB_LICMODE=onlinelicensing
fi

lic_args=()
if [[ -n "${MATLAB_LICMODE:-}" ]]; then
  lic_args=(-licmode "${MATLAB_LICMODE}")
fi

if [[ "${MATLAB_LICMODE:-}" == "onlinelicensing" && -n "${MATLAB_USER_ID:-}" && -n "${MATLAB_PASSWORD:-}" ]]; then
  "${script_dir}/login_online.bash"
fi

matlab_quote() {
  printf "%s" "$1" | sed "s/'/''/g"
}

script_dir_m="$(matlab_quote "${script_dir}")"
case_name_m="$(matlab_quote "${case_name}")"
output_path_m="$(matlab_quote "${output_path}")"

exec "${matlab_bin}" "${lic_args[@]}" -batch \
  "addpath('${script_dir_m}'); run_matpower_case('${case_name_m}', '${output_path_m}');"
