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
cases="${1:-${MATPOWER_SWEEP_CASES:-}}"
output_path="${2:-${MATPOWER_SWEEP_CSV:-results/matpower_ac_nr_lin_solver_sweep.csv}}"
solvers="${3:-${MATPOWER_LIN_SOLVERS:-}}"

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
cases_m="$(matlab_quote "${cases}")"
solvers_m="$(matlab_quote "${solvers}")"
output_path_m="$(matlab_quote "${output_path}")"

cd "${repo_root}"
exec "${matlab_bin}" "${lic_args[@]}" -batch \
  "addpath('${script_dir_m}'); sweep_nr_lin_solvers('${cases_m}', '${solvers_m}', '${output_path_m}');"
