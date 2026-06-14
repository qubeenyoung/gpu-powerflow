# ⚠️ 정밀도 정정 — "factorize fp32" 의 두 의미와 진짜 fp32 결과

> **상태**: 완료(정정)   **날짜**: 2026-06-12
> **한 줄**: 초기 측정은 `--single-precision fp32`(= fp32 *입력*, **fp64 factor**)를 썼다 — ncu 가 `factor_mid_blocked<double>` 을 노출해 발각. 진짜 fp32 factor 는 `--precision fp32`(`factor_mid_blocked<float>`)이고, 25K 가 1.85ms→**0.73ms**(≈2.5×)다. **진짜 fp32 의 ordering headroom 은 대형 case 에서 더 작다**(parallel-ND default 가 이미 near-optimal).

## 1. 무엇이 잘못됐나

`custom_linear_solver_run` 의 플래그:
- `--single-precision fp64|fp32` → **입력 값**의 저장 정밀도(`matrix_view.value_type`)만 바꾼다. factor 정밀도 아님.
- `--precision fp64|fp32|tf32` → `SolverConfig.precision` = **factor 정밀도**. **기본 FP64**.

→ `--single-precision fp32` 단독 = fp32 입력 + **FP64 factor**(config.precision 기본값). ncu 확인: `factor_mid_blocked<double,0>`, `factor_big<double>`. 01-findings 의 초기 수치(25K 1.85ms 등)는 **fp64 factor**였다.

진짜 fp32 factor = `--precision fp32` (필요시 `--single-precision fp32` 로 입력도 fp32). 커널 `factor_mid_blocked<float,0>`.

| 25K, B=1 | factorize_ms | relres | kernel |
|---|---:|---:|---|
| `--single-precision fp32` (fp32 in, **fp64 factor**) | 1.853 | 9.2e-5 | `<double>` |
| **`--precision fp32`** (**fp32 factor**) | **0.726** | 1.5e-4 | `<float>` |
| `--precision tf32` (TC) | 0.547 | 5.7e-2 | `<float,TC>` |

## 2. 진짜 fp32 factor 결과 (B=1, `--precision fp32 --single-precision fp32`)

**baseline = parallel-ND default median(5-run)**, ordering = best-of-k(`CLS_ORDER_K`, tail_cube):

| case | fp32 median (range) | best serial seed | **best-of-k** | speedup |
|---|---:|---:|---:|---:|
| 13K (8387) | 0.352 (0.313–0.360) | 0.299 (s42) | 0.309 (s41) | **1.14×** |
| 25K | 0.744 (0.706–0.771) | 0.707 (s7) | 0.708 (s7) | **1.05×** |
| 70K (USA) | 2.202 (2.119–2.284) | 2.038 (s44) | 2.115 (s7) | **1.04×** |

→ fp64 factor(01-findings, 1.12–1.15×)보다 **대형 case headroom 이 작다**. 이유: fp32 front 는 바이트가 절반(shared 점유↓ → occupancy↑) + 2× throughput → under-fill latency 비중↓ → **parallel-ND default 가 이미 near-optimal**(25K median 0.744 vs serial best 0.707, 5% 차). ordering 으로 줄일 여지가 작다.

## 3. 진짜 fp32 occupancy — ncu (graph-off build, `factor_mid_blocked<float>`)

25K deep level(seed44) 측정:

| level grid | block | warps_active | sm__throughput | dram |
|---:|---:|---:|---:|---:|
| 29 | 512 | **33.1%** | 6.6% | 4.1% |
| 12 | 512 | 33.1% | 3.3% | 1.6% |
| 6 | 512 | 33.1% | 2.2% | 1.2% |
| 5 | 512 | 33.1% | 2.0% | 0.9% |

**두 가지 occupancy 손실**: (a) grid<82 → idle SM, (b) block=512 thread=16 warp = **active SM 의 33%만**. dram 1–4% → **메모리 아님, latency 바운드**.

**(b) 검증 — 실패**: under-fill level 의 thread 를 512→768→1024 로 올려도(`CLS_MID_UNDERFILL_THREADS`) **0%**(8387/25k/USA 모두 ±0.2%). front 인수분해는 **panel-LU 의 직렬 pivot chain 에 latency 바운드** — warp 를 더 줘도 *한 front 내부* critical path 를 못 숨긴다. (b) 는 레버가 아니다. mid.cuh 주석에 기록.

**big 커널은 이미 fp32 1024 thread**(big.cuh:337) → 손댈 것 없음.

## 4. 진짜 fp32 negative results (01-findings 와 동일 구조벽 재확인)

| 레버 | fp32 결과 |
|---|---|
| under-fill thread 512→1024 | ✗ 0% (latency-bound per front) |
| multistream on/off | ◑ 이미 ON: 8387 1.03×, 25k 1.04×, **USA 1.15×**(big front overlap) |
| subtree stream 8→16 | ✗ (fp64 와 동일, 무효) |
| amalgamation (panel width) | ✗ (fp64 와 동일, 무효) |
| best-of-k ordering | ✅ 13K 1.14×, 25K 1.05×, 70K 1.04× |

## 5. 결론 — fp32 에서 1.2× 는?

- **소형(13K/8387)만 ~1.14× 근접** (ordering). 대형(25K/70K)은 **1.04–1.05× 천장** — parallel-ND default 가 이미 최적에 가깝고 occupancy 레버가 전무.
- fp64 factor 해석(01-findings)에서는 25K 가 1.15–1.18× 로 가장 높았다. **어느 정밀도 해석에서도 세 case 일괄 1.2× 는 구조적으로 불가**(B=1 under-fill 벽 + ordering 천장 < 1.2× 대형).
- **정직성**: worst-seed(25K fp64 2.69 / fp32 0.895) 대비로 1.2×+ 를 만들 수 있으나, 이는 notes 54/55 가 금지한 cap/baseline-inflation. parallel-ND median 대비가 정직한 비교.

## 재현
```bash
BIN=build/custom_linear_solver_run; C=/datasets/power_system/nr_linear_systems/case_ACTIVSg25k
# 진짜 fp32 factor baseline
$BIN $C --precision fp32 --single-precision fp32 --repeat 25 --warmup 8           # ~0.73ms (median, parallel ND)
# best-of-k ordering (deterministic)
CLS_ORDER_K=12 $BIN $C --precision fp32 --single-precision fp32 --repeat 30 --warmup 10 --metis-seed 1
# ncu 정밀도 확인
ncu --launch-skip 35 --launch-count 4 --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
  build-prof/custom_linear_solver_run $C --precision fp32 --no-multistream --repeat 1 --warmup 0 --serial-nd
```
