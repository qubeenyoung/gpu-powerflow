#!/usr/bin/env bash
set -euo pipefail

package_spec="${CUDSS_PIP_SPEC:-nvidia-cudss-cu12==0.7.1.6}"
install_root="${CUDSS_DIR:-/opt/nvidia/cudss}"
install_deps="${CUDSS_PIP_INSTALL_DEPS:-1}"

pip_args=(install --no-cache-dir)

if [[ "${install_deps}" == "0" ]]; then
    pip_args+=(--no-deps)
fi

python3 -m pip "${pip_args[@]}" "${package_spec}"

package_root="$(python3 - <<'PY'
from importlib import metadata
from pathlib import Path

def find_cudss_distribution():
    candidates = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name", "").lower().replace("_", "-")
        if name.startswith("nvidia-cudss-"):
            candidates.append(dist)
    if not candidates:
        raise SystemExit("Unable to locate an installed nvidia-cudss package")
    return sorted(candidates, key=lambda item: item.metadata.get("Name", ""))[-1]

dist = find_cudss_distribution()
root = None
for item in dist.files or []:
    if item.parts[-2:] == ("include", "cudss.h"):
        root = Path(dist.locate_file(item)).resolve().parents[1]
        break
if root is None:
    raise SystemExit("Unable to locate cuDSS wheel root from installed metadata")

if not (root / "include" / "cudss.h").is_file():
    raise SystemExit(f"Unable to locate cuDSS headers under {root}")
if not (root / "lib" / "libcudss.so.0").is_file():
    raise SystemExit(f"Unable to locate cuDSS libraries under {root}")

print(root)
PY
)"

mkdir -p "${install_root}"
rm -rf "${install_root}/include" "${install_root}/lib" "${install_root}/licenses"

cp -a "${package_root}/include" "${install_root}/"
cp -a "${package_root}/lib" "${install_root}/"

if [[ -d "${package_root}/licenses" ]]; then
    cp -a "${package_root}/licenses" "${install_root}/"
fi

python3 - <<PY
from importlib import metadata
from pathlib import Path
import shutil

def find_cudss_distribution():
    candidates = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name", "").lower().replace("_", "-")
        if name.startswith("nvidia-cudss-"):
            candidates.append(dist)
    if not candidates:
        raise SystemExit("Unable to locate an installed nvidia-cudss package")
    return sorted(candidates, key=lambda item: item.metadata.get("Name", ""))[-1]

dist = find_cudss_distribution()
for item in dist.files or []:
    parts = item.parts
    if len(parts) >= 3 and parts[0].startswith("nvidia_cudss_") and parts[1] == "licenses":
        src = Path(dist.locate_file(item)).resolve()
        dst = Path("${install_root}") / "licenses" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
PY

find "${install_root}/lib" -maxdepth 1 -type f -name '*.so.0' -print0 \
    | while IFS= read -r -d '' library; do
        link="${library%.0}"
        if [[ ! -e "${link}" ]]; then
            ln -s "$(basename "${library}")" "${link}"
        fi
    done
