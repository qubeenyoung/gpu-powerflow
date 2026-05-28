#!/usr/bin/env bash
# FP64 micro-lever sweep — RUN ONLY AFTER the GPU clock is locked from the HOST:
#   (host)  sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc 1395,1395
# Locking pins the core clock so cross-PROCESS bench runs are comparable (each
# gpu_mf_bench is a separate process; unlocked, the GPU idles to P8/210MHz between
# runs and re-boosts variably -> the cy125 "256 beats 512" measurement ARTIFACT).
# With the clock locked, MF_FT/MF_ST block-size sweeps are finally trustworthy.
# Methodology guard: only deltas >5% vs the 512/64 default are reported as real
# (sub-5% stays in the noise floor even locked); never claim a <5% "win".
set -u
BUILD=${BUILD:-/tmp/profile-build}
BENCH="$BUILD/gpu_mf_bench"

# --- gate on the lock being present (idle clock pinned high, not 210MHz P8) ---
gr=$(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -z "$gr" ] || [ "$gr" -lt 800 ] 2>/dev/null; then
  echo "ABORT: GPU clock not locked (idle clocks.gr=${gr:-?}MHz, expected ~1395)."
  echo "  On the HOST run:  sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc 1395,1395"
  echo "  Verify here:      nvidia-smi -q -d CLOCK"
  exit 1
fi
echo "Clock locked at ${gr}MHz — sweep is valid. bench=$BENCH"
echo

run_F() { MF_FT="$1" "$BENCH" 2>/dev/null | awk '/^case/{printf "%-18s %s\n",$1,$5}'; }
run_S() { MF_ST="$1" "$BENCH" 2>/dev/null | awk '/^case/{printf "%-18s %s\n",$1,$6}'; }

echo "=== FACTOR block-size sweep (MF_FT) — default 512 ==="
for ft in 256 384 512 640; do echo "--- MF_FT=$ft ---"; run_F "$ft"; done
echo
echo "=== SOLVE block-size sweep (MF_ST) — default per-level (64 power-grid) ==="
for st in 64 128 256; do echo "--- MF_ST=$st ---"; run_S "$st"; done
echo
echo "Compare the F=/S= columns across block sizes. ACCEPT a new default ONLY if it"
echo "beats 512/64 by >5% on a matrix WITHOUT regressing others >5%. Then change the"
echo "per-level default in gpu_mf.cu, re-run gpu_test (must stay 4/4), and commit."
