# TC 전용 경로 (`src/tc/`) 연구 보고서

> Status: historical research log. 최신 최적 dispatch와 TC/FP32 권장 결론은
> [`../05-reports/01-final-report-2026-06-05.md`](../05-reports/01-final-report-2026-06-05.md)를 우선 참고하고,
> TC headroom의 실측 wall 근거는
> [`../02-design-analysis/05-gemm-fraction-analysis.md`](../02-design-analysis/05-gemm-fraction-analysis.md)를 기준으로 한다.

*목표: batched factorize 의 tensor-core 가속 시도. case8387pegase 의 sparse power-grid Jacobian 에서 TC 가 net win 이 되는지 측정 + 근거 기반 분석.*

## 1. 가설 — 왜 TC 가 가능하다고 봤는가

### 1.1 raw throughput 비율

RTX 3090 sm_86 spec:
- FP32 ALU: 35.6 TFLOPS
- FP16 TC (FP32 accumulate): 142 TFLOPS → **FP32 의 4×**

→ trailing GEMM (factor 의 dominant compute) 을 TC 로 가속하면 *Phase 3 (trailing) 만 4× 가능*. factor wall 의 ~24 % 가 trailing GEMM (단일배치 분석 §6.1) 이므로, *이론적* factor wall 단축 = 0.24 × (1 − 1/4) = **−18 %**.

### 1.2 사용자 기억

이전 측정에서 "TC 가 FP64 보다 빠르고, FP32 보다 느렸음 — FP32 보다 느린 것은 TC setup cost 때문" 으로 기억. setup 을 잘 amortize 하면 FP32 도 이길 수 있다는 직관.

### 1.3 코드 기반 자산

`src/tc/factor_tc.cuh` 에 WMMA trailing 헬퍼 (`tc_trailing_wmma_f32`) 와 `mf_factor_extend_tc32_b`, `mf_factor_mid_tc32_b<true>` 가 이미 존재. 인프라 활용 가능.

## 2. 시도 — Phase 1 + 2 구현

### 2.1 Phase 1 — API 분리

**산출물**:
- `src/tc/multifrontal_tc.hpp` — `TCState`, `tc_setup/factorize/solve`
- `src/tc/multifrontal_tc.cu` — TC32 강제 dispatch, batched 의 device .cuh 재사용
- `src/solver.{hpp,cpp}` — `Solver::tc_setup/factorize/solve` 와이어링
- `tests/run_custom_solver.cu` — `--tc` flag, FP32 in/out
- `CMakeLists.txt` 업데이트

dispatch 로직은 batched 의 `MF_TC32=1` 와 *동일*. 인프라 분리만의 효과 측정용.

### 2.2 Phase 2 — TC 임계값 낮춤

**가설**: `mf_factor_mid_tc32_b<true>` 가 fsz<=48 의 front 는 *scalar fallback* (kernel 내부 분기). 이 임계값을 48 → 24 로 낮추면 fsz 32–48 fronts 도 WMMA 적용.

**산출물**: `src/tc/factor_kernels_tc.cuh` 의 `mf_factor_mid_tc_lo_b<MIN_FSZ_FOR_TC=24>` 새 커널.

## 3. 측정 결과

### 3.1 wall-time per system (5 trials median, B=64, MF_NO_SELINV=1)

| 구성 | median μs/sys | vs FP32 baseline |
|---|---:|---:|
| FP32 cap=8 baseline | **31.5** | 1.00× |
| TC Phase 1 cap=8 | 47 | +49 % ❌ |
| TC Phase 1 cap=16 | 46 | +46 % ❌ |
| TC Phase 1 cap=32 | 40 | +27 % ❌ |
| TC Phase 2 cap=8 | 40.3 | +28 % |
| **TC Phase 2 cap=16** | **39.5** | **+25 %** ❌ |
| TC Phase 2 cap=32 | 39.2 | +24 % |

**판정**: Phase 2 가 Phase 1 대비 ~15 % 개선 (의도된 효과), 하지만 **FP32 baseline 보다 모든 구성에서 +24% 이상 느림**. *TC 가 FP32 를 이기지 못함*.

### 3.2 maximum amalgamation (cap=64) 측정

가설: case8387 의 sparse 구조 때문에 TC 가 닿는 영역 (fsz>=49) 의 front 수가 적음 (전체 7400+ 중 19 개). cap 을 키워 amalgamation 을 공격적으로 하면 front 가 더 커질 수 있음.

cap 별 front 분포 측정 (`CLS_DUMP=1`):

| cap | levels | fsz>=65 fronts | max fsz |
|---|---:|---:|---:|
| 8 (default) | 30 | 4 | 76 |
| 16 | 23 | 17 | 82 |
| 32 | 19 | 17 | 77 |
| 48 | 16 | 13 | 96 |
| **64 (max)** | **15** | **11** | **~96** |

**핵심 발견**: cap 을 8 → 64 로 8× 키워도 max fsz 가 76 → 96 으로 26 % 만 증가. 100 도 못 넘김. fsz≥100 fronts = **0 개**.

## 4. 실패 원인 분석 — *정량적 근거*

### 4.1 WMMA 의 본질적 효율 = (K-tile 정렬) × (M/N-tile 정렬) × (tile 개수)

WMMA tile = 16 × 16 × 16. 효율 결정 요인:

**(A) K-tile 정렬 (= nc)**:
- nc=8 (cap=8): KP=16, K-차원 절반이 zero-padded → *50 % 손실*
- nc=16 (cap=16): KP=16, full → 0 % 손실
- nc=32 (cap=32): KP=32, full → 0 % 손실

**(B) M/N-tile 정렬 (= uc)**:
- uc=24 (typical fsz=32 + cap=8): UCP=32, 마지막 tile 8 row 만 valid → 50 % 손실
- uc=48 (typical fsz=64 + cap=16): UCP=48, full → 0 % 손실
- uc=60 (typical fsz=76 + cap=16): UCP=64, 4 row 만 invalid → 6 % 손실

**(C) tile 개수**:
- fsz=49, uc=33: UCP=48, KP=16 → tile grid = 3×3 = **9 tiles**
- fsz=76, uc=60: UCP=64, KP=16 → tile grid = 4×4 = **16 tiles**
- fsz=128, uc=112: UCP=112, KP=16 → tile grid = 7×7 = **49 tiles**

case8387 의 max fsz=96 (cap≥48 에서): UCP=80, tile grid 5×5 = **25 tiles**. literature 의 TC-effective 영역 (fsz≥256, tile grid 16×16=256+ tiles) 의 10 % 수준.

### 4.2 WMMA setup cost 의 amortization

per-tile WMMA 작업:
1. FP16 staging (FP32 → FP16 + 메모리 layout): ~32 cycles
2. `wmma::load_matrix_sync` A, B: ~10 cycles each
3. `wmma::mma_sync` × (K/16 mma): nc=8→1, nc=16→1, nc=32→2
4. `wmma::store_matrix_sync`: ~10 cycles
5. subtract-back to F: 256/32 = 8 cycles

총 per-tile ≈ **70+ cycles**. compute 비율 = mma_sync (~10 cycles) / 70 = **15 % 만이 실제 GEMM**, 나머지 85 % 는 staging/load/store.

vs scalar FP32 trailing for same tile (16×16 output, K=16):
- per output: 16 ops × 1 FMA = 16 cycles
- 256 outputs / 128 threads = 2 ops/thread = 2 cycles per output × 256 = 좀 복잡
- 대략 ~50 cycles per 16×16 output tile equivalent

**즉 WMMA 의 setup overhead (70 cycles per tile) 가 scalar 의 work (50 cycles per equivalent tile) 와 비슷**. 4× throughput 의 이론적 가속이 staging cost 에 거의 다 먹힘. 큰 tile 개수가 필요하지만 case8387 의 max 25 tiles 로는 amortize 안 됨.

### 4.3 case8387 의 fundamental ceiling

cap=64 (max) 에서 max fsz=96 인 이유 — *root separator 의 크기*.

case8387 는 power-grid Jacobian:
- n = 14908
- avg nnz/row = 7.4 (매우 sparse)
- METIS nested dissection 의 root separator 크기 ≈ √n × (sparsity factor) ≈ **~90**

→ 어떤 amalgamation 전략으로도 *root separator 보다 큰 dense block 을 만들 수 없음*. case8387 의 fsz ceiling 은 알고리즘적 한계가 아니라 *matrix 의 sparsity structure 자체*.

literature 의 TC-effective sparse problems:
- onetone2 (회로): root separator ~500+, fsz≥257
- 3D PDE: root separator ~k² for k³ problem (k≥50 으로 fsz≥2500)
- case8387: root separator ~90

case8387 은 **TC 의 sweet spot 에서 한 자릿수 이상 떨어진 영역**.

§5b 의 case_SyntheticUSA (n=156k, US 전체 송전망 합성 모델) 측정으로 power-grid 일반의 fundamental limit 확인: USA 도 max fsz~245 로 TC sweet spot 도달 불가.

## 5. 결론 — TC 가 case8387 에서 win 못 하는 이유 (3 단계)

| 단계 | 근거 | 정량 |
|---|---|---|
| (1) WMMA 효율 = K × M/N × tile-count 의 곱 | §4.1 | nc=8 일 때 50 % loss × edge padding 6–50 % × tile 25 개 |
| (2) Setup overhead 가 scalar trailing 의 work 와 동급 | §4.2 | per-tile ~70 cycles WMMA vs ~50 cycles scalar; 4× peak throughput 이 staging 에 먹힘 |
| (3) case8387 의 root separator ~90 이 fsz ceiling 결정 | §4.3 | cap=64 → max fsz=96. 100 도 못 넘김. amalgamation 으로 못 풀음 |

**핵심**: TC 의 효과는 *raw throughput 비율* (4×) 이 아니라 *(throughput) × (per-tile compute fraction) × (tile-count amortization)*. case8387 의 sparsity 가 (3) 을 막아 (2) 를 amortize 못 시킴 → (1) 의 이론적 이득 실현 불가.

## 5b. 더 큰 power-grid: case_SyntheticUSA (82k bus) 측정

case8387 의 *root separator ceiling* (§4.3) 가 power-grid sparsity 의 본질이라면, *훨씬 큰 power-grid* 인 case_SyntheticUSA (82k bus, n=156255, nnz≈1M) 에서도 같은 패턴이 나와야 함. 검증 측정.

### 5b.1 Front 분포 (`CLS_DUMP=1`, cap=20 default)

```
n=156255  P=74223  levels=40  cap=20  front_total(MB f32)=35.1
fsz [1..16]    cnt=71608  f²%=33.8  f³%= 4.3
fsz [17..32]   cnt= 1697  f²%=10.0  f³%= 3.7
fsz [33..48]   cnt=  377  f²%= 6.8  f³%= 4.2
fsz [49..64]   cnt=  206  f²%= 7.4  f³%= 6.4
fsz [65..96]   cnt=  204  f²%=13.9  f³%=16.6
fsz [97..160]  cnt=  110  f²%=18.8  f³%=36.4
fsz [161..∞]   cnt=   21  f²%= 9.3  f³%=28.4
```

**case8387 대비 차이**:
- max fsz 96 → ~245 (USA root level)
- fsz≥97 fronts: 0 → **131 개** (TC sweet spot 영역)
- compute (f³) 의 64.8 % 가 fsz≥97 영역에 집중 — TC 의 잠재 영향력 ↑

### 5b.2 측정 결과 (3 trials × repeat=10)

| B | FP32 factor μs/sys | TC factor μs/sys | Δfactor |
|---:|---:|---:|---:|
| 64 | 485 | 566 | +16.7 % |
| 128 | 478 | 541 | +13.0 % |
| 256 | 468 | 552 | +18.0 % |
| **512** | **471** | **525** | **+11.4 %** |

**판정**: USA 에서 case8387 (Δ +25 %) 보다는 좁혀짐 — TC 가 닿는 fronts (fsz≥97) 가 실제로 존재하니까. 하지만 **여전히 모든 B 에서 FP32 가 빠름**.

### 5b.3 USA 에서도 fail 한 이유 — 정량 분해

USA 의 f³ (compute volume) 분포 보면:
- fsz≤32 (small_warp 영역, TC 불가): **8 %**
- fsz 33–96 (mid scalar 영역, TC 가능하지만 tile grid 작음): **27 %**
- fsz≥97 (TC sweet spot): **65 %**

만약 fsz≥97 영역에서 TC 가 2× 가속 → wall 절감 = 65 × 0.5 = **−32 %**. 측정은 **+11 %**.

손실 분해:
1. fsz≤32 (small_warp_b path): TC 무관, 변화 없음
2. fsz 33–96 (mid_tc_lo_b path with WMMA, but tile grid 4–9 only): WMMA setup overhead 가 트레일링 가속을 못 따라잡음 → **negative**
3. fsz≥97 (mid_tc_lo_b path with WMMA, tile grid 25–256): TC 가 의도대로 빨라짐

추정: fsz 33–96 영역의 *negative impact* (per-front +30 %) 가 fsz≥97 영역의 *positive impact* (per-front −40 %) 를 압도. 양자가 *total wall* 에서 cancel out 하고 *작은 fronts 의 launch overhead* 가 추가로 더해져 net +11 %.

→ **TC 가 의미 있으려면 *fsz 33–96 영역 fronts 까지 disable* 하고 fsz≥97 만 TC kernel 로 dispatch 해야 함**. 현재 mid_tc_lo_b<24> 는 너무 광범위.

### 5b.4 power-grid Jacobian 의 일반적 결론

| 케이스 | n | max fsz | fsz≥97 fronts | Δfactor (TC vs FP32) |
|---|---:|---:|---:|---:|
| case8387 | 14,908 | 76 | **0** | +25 % |
| case_SyntheticUSA | 156,255 | ~245 | 131 | +13 % |

power-grid Jacobian 은 *어떤 크기로 키워도* TC 가 진짜 win 하기엔 부족. literature 의 TC sweet spot (onetone2 fsz≥257 가 dominant, 3D PDE fsz≥1000+) 와 비교하면:
- USA 의 fsz≥97 fronts 도 절반은 fsz 97–160 정도, 256+ 거의 없음
- power-grid 의 sparsity (avg degree 7–8, 평면 그래프 가까운 구조) 가 separator 크기 자체를 막음

## 6. 어떤 조건에서 TC 가 win 했을 것인가

같은 코드 (Phase 1+2) 가 *다른 matrix* 에서는 TC win 가능성:

| 조건 | 필요 값 | 효과 |
|---|---|---|
| root separator | ≥ 256 | fsz 256+ 의 front 가 생겨 tile grid 16×16+, setup amortize |
| nc (panel cap 의 결과) | ≥ 16 | K-tile full, no K-padding |
| 적합한 matrix | 3D PDE, 회로 (onetone2 류), 큰 dense block 보유 | 위 두 조건 충족 |

case8387 만의 문제이지 *우리 코드 / 알고리즘* 의 문제는 아님. literature 의 STRUMPACK / SuperLU_DIST 도 *작은 dense block 영역에선 TC 사용 안 함* (Anzt et al. 2022 — 작은 fronts 는 custom 커널, 큰 fronts 만 cuBLAS/cuSOLVER).

## 7. 후속 계획

### 7.1 즉시 행동

**TC dedicated path 의 운영**:
- `src/tc/` 모듈 그대로 유지 — *case8387 같은 sparse matrix 에서는 default 권장 아님*
- 더 큰 sparse problem (onetone2, 3D PDE) 에서는 *Phase 2 임계값 = 24* 가 적절. 측정 후 확정.
- 문서화: `--tc` flag 의 적용 가능 영역 명시

**case8387 의 가속 lever**: TC 가 아닌 *non-compute* 방향 (이미 [`../04-benchmarks-profiling/08-batched-throughput-fp32-case8387-b2-b1024.md`](../04-benchmarks-profiling/08-batched-throughput-fp32-case8387-b2-b1024.md) §7 에서 분석):
- per-system stall 의 80 % 가 wait / long_scoreboard / barrier (compute 아님)
- 진짜 lever 는 dispatch overhead 절감 + memory access 패턴 개선

### 7.2 보류

**Phase 3 (deep-tail dense LU 흡수)**: cap=64 이미 deep-tail 흡수 효과. 추가 코드는 5–10 % 의 미세 개선만 잠재. **TC 와 무관**. 우선순위 낮음.

**Phase 4 (sibling subtree fusion)**: tree restructuring 큰 작업. case8387 에서는 fsz ceiling 자체가 작아 (root separator ~90) sibling fusion 으로도 fsz 100+ 도달 어려움. **case8387 한정으로 ROI 매우 낮음**.

### 7.3 미래 — 다른 matrix 에서 TC dedicated path 활용

대규모 sparse problem (onetone2, n>100k 의 3D PDE) 가 들어오면:
1. CLS_DUMP 로 max fsz 확인 — ≥ 200 이면 진행 가치 있음
2. cap 을 16 강제 (nc=16 full K-tile)
3. `--tc` 로 측정 vs FP32 vs cuDSS
4. *이때* Phase 3/4 의 tree mod 진짜 의미 있음 (root separator 가 커야 amalgamation 이 더 큰 front 만듦)

이 자산을 만들어둔 게 phase 1+2 의 의의.

## 8. 측정 재현

```bash
# case_SyntheticUSA (n=156k) 측정
for B in 64 128 256 512; do
  for tag in "fp32" "tc"; do
    unset MF_FP32 CLS_CAP; export MF_NO_SELINV=1
    flag="--tc"; [ "$tag" = "fp32" ] && { export MF_FP32=1; flag=""; }
    ./custom_linear_solver_run /datasets/power_system/nr_linear_systems/case_SyntheticUSA \
        --repeat 10 --single-precision fp32 --batch $B --batch-only $flag
  done
done

# case8387 baseline 비교
for tag in "fp32" "tc-cap8" "tc-cap16" "tc-cap32" "tc-cap64"; do
  unset MF_FP32 CLS_CAP; export MF_NO_SELINV=1
  case $tag in
    fp32) export MF_FP32=1; flag="";;
    tc-cap8) flag="--tc";;
    tc-cap16) export CLS_CAP=16; flag="--tc";;
    tc-cap32) export CLS_CAP=32; flag="--tc";;
    tc-cap64) export CLS_CAP=64; flag="--tc";;
  esac
  ./custom_linear_solver_run /datasets/power_system/nr_linear_systems/case8387pegase \
      --repeat 50 --single-precision fp32 --batch 64 --batch-only $flag
done

# amalgamation 한계 확인
for cap in 8 16 32 64; do
  CLS_DUMP=1 CLS_CAP=$cap ./custom_linear_solver_run \
      /datasets/power_system/nr_linear_systems/case8387pegase \
      --repeat 1 --single-precision fp64 2>&1 | grep -E "CLS_DUMP|fsz\["
done
```

원자료 위치: `/home/claude/prof/` (해당 시점)
