#!/usr/bin/env bash
set -euo pipefail

# Small human-facing wrapper around run_matpower_case.m.
#
# The benchmark package calls the .m file directly from Python. This shell
# wrapper is kept for manual smoke checks, container validation, and quick
# MATPOWER license/path debugging.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

# Load repository-local settings without requiring users to export every value
# before running the script. The .env file is git-ignored and may contain local
# MATLAB/MATPOWER paths or online licensing credentials.
for env_file in "${repo_root}/.env" "${script_dir}/.env"; do
  if [[ -f "${env_file}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
  fi
done

matlab_bin="${MATLAB_BIN:-matlab}"
case_name="${1:-${MATPOWER_CASE:-case9}}"
output_path="${2:-${MATPOWER_RESULT_JSON:-}}"

# MATLAB accepts network license locations through MLM_LICENSE_FILE.  The
# repository also recognizes MATLAB_LICENSE_FILE because that name is easier to
# read in .env files; map it once if the MATLAB-native variable is unset.
if [[ -n "${MATLAB_LICENSE_FILE:-}" && -z "${MLM_LICENSE_FILE:-}" ]]; then
  export MLM_LICENSE_FILE="${MATLAB_LICENSE_FILE}"
fi

export MATPOWER_HOME="${MATPOWER_HOME:-/opt/matpower}"

# If credentials are present and no license server/file is configured, prefer
# MathWorks online licensing. login_online.bash performs the non-interactive
# authentication handshake before MATLAB is launched for the actual run.
if [[ -z "${MATLAB_LICMODE:-}" && -n "${MATLAB_USER_ID:-}" && -n "${MATLAB_PASSWORD:-}" && -z "${MLM_LICENSE_FILE:-}" ]]; then
  MATLAB_LICMODE=onlinelicensing
fi

lic_args=()
if [[ -n "${MATLAB_LICMODE:-}" ]]; then
  lic_args=(-licmode "${MATLAB_LICMODE}")
fi

if [[ "${MATLAB_LICMODE:-}" == "onlinelicensing" && -n "${MATLAB_USER_ID:-}" && -n "${MATLAB_PASSWORD:-}" ]]; then
  "${script_dir}/login_online.bash"
fi

# MATLAB string literals escape a single quote by doubling it.  Keep this helper
# tiny and explicit so paths with spaces or apostrophes still survive -batch.
matlab_quote() {
  printf "%s" "$1" | sed "s/'/''/g"
}

script_dir_m="$(matlab_quote "${script_dir}")"
case_name_m="$(matlab_quote "${case_name}")"
output_path_m="$(matlab_quote "${output_path}")"

exec "${matlab_bin}" "${lic_args[@]}" -batch \
  "addpath('${script_dir_m}'); run_matpower_case('${case_name_m}', '${output_path_m}', 'Mode', 'single');"
