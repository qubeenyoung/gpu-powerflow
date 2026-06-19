#!/usr/bin/env bash
# Fetch SuiteSparse matrices used by the out-of-domain generalization benchmarks.
# These are large and git-ignored (datasets/suitesparse/), so they live here only
# after this script downloads them on demand.
#
# Source: SuiteSparse Matrix Collection, https://sparse.tamu.edu/MM/<group>/<name>.tar.gz
# Each tarball extracts to <name>/<name>.mtx (the collection's convention).
#
# Usage:
#   ./fetch_suitesparse.sh                 # default set (cant parabolic_fem ...)
#   ./fetch_suitesparse.sh Serena G3_circuit   # specific matrices by name
#   ./fetch_suitesparse.sh --all           # every known matrix below
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/suitesparse"
mkdir -p "$DEST/downloads"

# name -> SuiteSparse group.
declare -A GROUP=(
  [cant]=Williams [bmwcra_1]=GHS_psdef [G3_circuit]=AMD [parabolic_fem]=Wissgott
  [scircuit]=Hamm [TSOPF_RS_b2383]=TSOPF
  [Serena]=Janna [Geo_1438]=Janna [Hook_1498]=Janna [ML_Geer]=Janna
  [Transport]=Janna [Flan_1565]=Janna [Cube_Coup_dt0]=Janna
)
DEFAULT=( cant parabolic_fem G3_circuit bmwcra_1 )

# Resolve the requested list.
if [ "${1:-}" = "--all" ]; then
  NAMES=( "${!GROUP[@]}" )
elif [ "$#" -gt 0 ]; then
  NAMES=( "$@" )
else
  NAMES=( "${DEFAULT[@]}" )
fi

for name in "${NAMES[@]}"; do
  group="${GROUP[$name]:-}"
  [ -n "$group" ] || { echo "unknown matrix '$name' (add it to GROUP[])"; continue; }

  mtx="$DEST/$name/$name.mtx"
  if [ -f "$mtx" ]; then echo "have   $name"; continue; fi

  # 1. Download the tarball (skip if already present).
  tar="$DEST/downloads/$name.tar.gz"
  url="https://sparse.tamu.edu/MM/$group/$name.tar.gz"
  [ -f "$tar" ] || { echo "fetch  $name <- $url"; curl -fL --retry 3 -o "$tar" "$url"; }

  # 2. Extract <name>/<name>.mtx into the dataset dir.
  echo "unpack $name"
  tar -xzf "$tar" -C "$DEST"
  [ -f "$mtx" ] || { echo "warning: $mtx missing after extract"; }
done

echo "done. matrices under: $DEST/<name>/<name>.mtx"
