#!/usr/bin/env bash
set -euo pipefail

python_packages="${POWER_PYTHON_PACKAGES:-pypower pandapower matpower matpowercaseframes}"
power_dataset_root="${POWER_SYSTEM_DATASET_ROOT:-/datasets/power_system}"
matpower_dataset_root="${MATPOWER_DATASET_ROOT:-${power_dataset_root}/matpower}"
matpower_ref="${MATPOWER_REF:-master}"
matpower_tmp="${MATPOWER_TMP:-/tmp/matpower-src}"

python_packages="${python_packages//,/ }"
read -r -a python_package_list <<< "${python_packages}"

python3 -m pip install --no-cache-dir --upgrade pip
python3 -m pip install --no-cache-dir "${python_package_list[@]}"
python3 - <<'PY'
import importlib

for name in ("pypower", "pandapower", "matpower", "matpowercaseframes"):
    importlib.import_module(name)
PY

rm -rf "${matpower_tmp}" "${matpower_dataset_root}"
git clone --depth 1 --branch "${matpower_ref}" https://github.com/MATPOWER/matpower.git "${matpower_tmp}"

mkdir -p "${matpower_dataset_root}"
rsync -a \
    --include='*/' \
    --include='*.m' \
    --exclude='*' \
    "${matpower_tmp}/data/" \
    "${matpower_dataset_root}/"

{
    printf 'MATPOWER power-system datasets\n'
    printf 'Source: https://github.com/MATPOWER/matpower.git\n'
    printf 'Ref: %s\n' "${matpower_ref}"
    printf 'Stored at: %s\n' "${matpower_dataset_root}"
    printf 'Copied pattern: data/**/*.m\n'
    printf 'Generated at: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'Case files:\n'
    find "${matpower_dataset_root}" -type f -name '*.m' -printf '%P\n' | sort
} > "${matpower_dataset_root}/MANIFEST.txt"

rm -rf "${matpower_tmp}"
