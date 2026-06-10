# 최적 구성 (Optimal Configuration)

**작성일**: 2026-06-10
**목적**: 현재 최적 경로의 빌드/런타임 설정을 한곳에. 메소드 자체의 설명은 [`storyline.md`](storyline.md),
세부 측정은 `03-optimization-notes/` 참조. 여기서 토글 이름 ↔ 메소드 매핑만 빠르게 본다.

---

## 1. 권장 기본 경로

- **정밀도/정확도**: `--precision tf32` (front=FP32, trailing=TF32 텐서코어) **+ Ozaki first-order 보정**.
  → relres 가 FP32 band(~1e-4..1e-5)이면서 fp32 대비 빠름.
- **배치**: `--batch 64`(latency) ~ `256`(throughput).
- **항상 ON(전제/스케줄링)**: no-pivot, CUDA graph, METIS-ND, 3단 라우팅, 멀티스트림, 동형 디스패치.

regime 에 따라 텐서코어 적용 방식만 갈린다.

### A. Large regime (n ≳ 25k: ACTIVSg25k / 70k / SyntheticUSA)

```bash
cmake -S custom_linear_solver -B build-large -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release \
  -DCLS_MID_TF32_TC=ON -DCLS_MID_TF32_DIRECT_SHARED=ON -DCLS_MID_TF32_LOW_TC=ON \
  -DCLS_MID_LOW_SPLIT=ON -DCLS_BIG_LOW_SPLIT=ON \
  -DCLS_BIG_TF32_BLOCKED_TC=ON -DCLS_BIG_TF32_SHARED_THREADS_512=ON \
  -DCLS_RESPECT_PANEL_CAP=ON -DCLS_MID_TF32_MIN_FSZ=48 \
  -DCLS_TF32_OZAKI_TC2_FIRST_ORDER=ON
cmake --build build-large -j

build-large/custom_linear_solver_run <case> --precision tf32 --batch 64 \
  --repeat 61 --warmup 8 --single-precision fp64
```

### B. Low-fill regime (n ≲ 16k: case8387pegase / case13659pegase)

A 의 플래그에 **TC 적합 패널 융합(M6)** 추가:

```bash
  ... (A 의 플래그) \
  -DCLS_TC_CLOSURE_PANEL_AMALGAMATE=ON -DCLS_TC_CLOSURE_PANEL_AMALGAMATE_CAP=32 \
  -DCLS_SMALL_FRONT_MAX_16=ON -DCLS_MID_TF32_TC_THREADS_128=ON -DCLS_MID_TF32_LOW_TC_FORCE_ALL=ON
# 실행은 결정성을 위해 --serial-nd 권장
build-lowfill/custom_linear_solver_run <case> --precision tf32 --batch 64 \
  --serial-nd --metis-seed 7 --panel-cap 30 --repeat 61 --warmup 8 --single-precision fp64
```

> low-fill 정책은 large 에 universal 하지 않다(25K 불안정). 두 정책은 분리 빌드로 유지한다.

---

## 2. 토글 ↔ 메소드 매핑 (ablation 참조표)

| 토글 (CMake -D / CLI) | 기본 | 메소드 (storyline) |
|---|---|---|
| `--precision {fp64,fp32,fp16,tf32}` | fp64 | substrate(정밀도 regime) / M5 활성 전제 |
| `--batch N` | 1 | M1 배치 front-major |
| `SolverConfig.tier_split` | true | M2 3단 커널 라우팅 |
| 동형 디스패치 정책 (band-order, baked) | gate | M3 동형 디스패치 |
| `--no-multistream` | (multi=on) | M4 서브트리 멀티스트림 |
| `CLS_MID_TF32_TC` | OFF | M5 TF32 텐서코어 trailing (mid) |
| `CLS_BIG_TF32_BLOCKED_TC` | OFF | M5 TF32 텐서코어 trailing (big) |
| `CLS_MID_LOW_SPLIT` / `CLS_BIG_LOW_SPLIT` | OFF | M5 보조 (dispatch bucketing) |
| `CLS_BIG_TF32_SHARED_THREADS_512` / `CLS_MID_TF32_TC_THREADS_128` | OFF | M5/M6 launch shape |
| `CLS_MID_TF32_MIN_FSZ` | 48 | M5 TC 적격 gate |
| `CLS_RESPECT_PANEL_CAP` | OFF | panel_cap 가시화 (M5 와 함께) |
| `CLS_TC_CLOSURE_PANEL_AMALGAMATE(_CAP)` | OFF | M6 TC 적합 패널 융합 |
| `CLS_SMALL_FRONT_MAX_16` | OFF | M6 tier 경계 조정 (`kSmallFrontMax 32→16`) |
| `CLS_TF32_OZAKI_TC2` / `..._FIRST_ORDER` | OFF | M7 Ozaki 오차보정 |
| `CLS_TF32_COLUMN_USOLVE` | OFF | (음성) column-U-solve — 효과 ≈0 |
| `CLS_INTERNAL_GRAPH` | ON | substrate(CUDA graph) |
| `CLS_USE_PIVOTING` | OFF | substrate(no-pivot 비교용) |

---

## 3. 검증된 수치 (sm_86, RTX 3090, 본 세션 재현)

| regime | case | B | fp32 ms/sys | tf32+Ozaki ms/sys | speedup | tf32+Ozaki relres |
|---|---|---:|---:|---:|---:|---:|
| low-fill | case8387pegase | 64 | 0.03318 | 0.02787 | **1.19×** | 4.77e-5 (FP32 band) |
| large | case_ACTIVSg25k | 64 | 0.11351 | 0.09484 | **1.20×** | 2.97e-4 |

(Ozaki 미적용 raw tf32 는 8387 B64 **1.24×** 지만 relres 3.97e-2; Ozaki 가 정확도를 FP32 band 로
회수하며 속도는 ~5%만 양보.)
