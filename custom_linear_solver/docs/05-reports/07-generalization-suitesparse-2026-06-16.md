# 일반화 실험 — SuiteSparse 행렬에서 STRUMPACK·cuDSS·우리 (2026-06-16)

> **상태**: canonical   **한 줄**: 우리 기여(tiny-front packing+fusion 입도)가 power-flow 특수현상이 아님을
> 다른 행렬류(circuit·2D-FEM)로 검증하고, 그 과정에서 FP64 large-tier 의 한도 버그를 고치고 multi-block 으로
> 최적화해 큰-front 2D 행렬에서도 cuDSS 와 경쟁/우위에 도달.

재현 하니스·데이터: `exp/harness/{strumpack_bench,cudss_bench}.cpp`, `exp/sym_to_full.py`
(행렬은 SuiteSparse MM, 로컬 `exp/cases_ss*` — repo 미포함). 본 솔버 novelty 맥락: [main-report §3](../main-report.md#3-핵심-기여--논리-구조).

---

## 1. 목적

main-report §3 의 기여는 "front < warp 인 tiny-front 레짐에서 packing+full-fusion 을 sub-group 입도로 동시
달성"이다. 이게 **power-flow 만의 현상인지, tiny-front 행렬류 일반인지**를 SuiteSparse 로 검증한다 — front 크기
스펙트럼(2D/회로 tiny-front → 3D large-front)을 가로질러 STRUMPACK+MAGMA·cuDSS·우리를 측정.

## 2. baseline 특성화 — tiny-front 바닥은 일반적이다 (RTX 3090, FP64)

factor = refactorization(= Newton step). STRUMPACK/cuDSS 배율 = launch/occupancy-bound multifrontal 의 손해.

| 행렬 | 류 | n | fill/row* | cuDSS f/s | STRUMPACK f/s | STRMPK/cuDSS (f, **s**) |
|---|---|---|---|---|---|---|
| scircuit | 회로(tiny) | 171k | 16 | 4.98 / 1.53 | 75.9 / 85.5 | 15× / **56×** |
| G3_circuit | 2D 회로 | 1.59M | 114 | 395 / 15.2 | 1851 / 1339 | 4.7× / **88×** |
| parabolic_fem | 2D FEM | 526k | 97 | 79.4 / 5.65 | 257 / 161 | 3.2× / **28×** |
| cant | 3D FEM | 62k | 588 | 63.2 / 3.02 | 150 / 17.7 | 2.4× / **5.9×** |
| bmwcra_1 | 3D 구조 | 149k | 881 | 520 / 9.95 | 617 / 49.0 | 1.2× / **4.9×** |

*fill/row = STRUMPACK factor nnz / n (front 크기 대용치). 작을수록 tiny-front.

**결론**: per-level vbatched 멀티프론탈(STRUMPACK)이 tiny-front(회로·2D-FEM)에서 solve 28–88× 느리고, 3D
large-front 에선 5–6× 로 소멸. → **tiny-front launch/occupancy 바닥은 power-flow 특수현상이 아니라 넓은
행렬류의 현상**(우리 기여의 적용 경계가 측정으로 확인됨). cuDSS(supernodal)는 두 레짐 다 견고 = 진짜 경쟁자.

## 3. 우리 솔버 적용 — 정정 2건 + 버그 1건 + 최적화 1건

**정정 1 — ordering 은 정확하다.** `predicted_fill`(우리 METIS perm) = STRUMPACK 과 일치(scircuit 2.59M≈2.7M,
parabolic 50.5M≈51M). 초기 "ordering 결함" 가설은 틀렸다.

**버그 — large-tier shared 한도 초과("OOM"은 오진).** large-tier 가 L/U 를 `2·nc·uc` 만큼 shared 에 올리는데
parabolic separator 의 uc(최대 643)에서 **164KB > 99KB 한도** 초과 → 커널 launch `cudaErrorInvalidValue` 가
caller 에서 "allocation failed"로 오표기(24GB 멀쩡, OOM 아님). 코드 주석이 *"power-grid Jacobian 에선 항상
budget 에 맞는다"* 고 가정을 박아놨던 게 원인.
→ **수정**: U 만 j-타일(폭 `nc·jt ≤ budget`)로 staging, L 은 row-contiguous 라 global 직접 읽기. 공통 power-grid
경로는 바이트 단위로 그대로. (`e953559`)

**FP64 분석 — 단일시스템은 under-fill 바닥.** ncu:

| | 지배 커널 | grid | SM 처리율 | 진단 |
|---|---|---|---|
| power-flow B=1 | factor_small 52.7% | 1–9 블록 | 0.5–3.4% | 상위 레벨 front 1–9개 → under-fill |
| parabolic | factor_large 98.6% | 2 블록 | 0.8% | under-fill + 거대 front 를 block=128 이 직렬 처리 |

→ B=1 단일시스템은 트리 상단에서 GPU 를 못 채우는 임계경로 바닥(tier 무관). 헤드룸은 배치(이미 활용)와 큰 front.

**최적화 — FP64 large-tier multi-block.** 큰 front 의 trailing(uc²·nc)은 embarrassingly parallel 인데 one-block-
per-front 라 starved. **2-커널 분리**: `factor_large_panel`(front당 1블록, LU+U-solve) + `factor_large_trail`
(3D grid (타일,front,배치), 32×32 타일을 front당 다중 블록으로 분산, L/U 는 panel 커널이 쓴 걸 global 에서 읽음,
contribution 은 부모로 fused). grid 가 GPU 를 채움. (`dd6f0f3`→`7678a6c`)
*리뷰로 잡은 버그*: grid.x 는 `level_max_uc`(unclamped)로 sizing 해야 함 — `max_uc`(TF32 캡 클램프)로 하면 256
너머 타일 누락 → 결과 손상(relres 11).

## 4. 결과 — 일반화 확인

**parabolic_fem (FP64), 우리 최적화 전후 vs 벤더:**

| | factor ms | solve ms | relres |
|---|---|---|---|
| 우리 (single-block 원본) | 759 | 7.2 | 1.7e-11 |
| **우리 (multi-block)** | **65.6** (11.5×↑) | 7.2 | 1.7e-11 |
| cuDSS | 79.4 | 5.7 | — |
| STRUMPACK | 257 | 161 | — |

⇒ **최적화 후 parabolic factor 에서 cuDSS 추월(65.6 vs 79.4)**, solve cuDSS 대등 + STRUMPACK 의 23×.

**scircuit (FP64):** B=1 우리 factor 4.4 / solve 2.2 ≈ cuDSS 5.0 / 1.5 (대등). 배치 B=64: per-system factor
0.90(B=1 대비 5.2×) / solve 0.24(8.5×) — 우리 배치 메커니즘이 회로 행렬에서도 작동. relres 3.5e-10(no-pivot OK).

**무회귀**: power-flow 25k FP64 factor 1.09 / solve 0.60 / relres 2.4e-14(불변), B=64 per-sys 0.074/0.032,
scircuit 불변. power-grid/scircuit 은 large-tier 미사용 → 위 변경 전부 byte-identical. TF32/FP32 large 경로 미변경.

## 4b. B=1 FP64 3-way (front 크기 스펙트럼) + 추가 최적화 반영

FP32/TF32 large-tier 도 multi-block 으로 통일(`1e78c7b`), batched 큰-front solve 크래시도 non-staged
fallback 으로 수정(`01d9d4d`). 최신 B=1 FP64(단일 시스템, warmed) 3-way:

| 행렬 | 레짐 | custom f/s | cuDSS f/s | STRUMPACK f/s | STR/cuDSS factor |
|---|---|---|---|---|---|
| scircuit | tiny-front | **4.39** / 2.18 | 4.98 / **1.53** | 77.9 / 87.3 | STR **15.6× 느림** |
| parabolic_fem | 2D-FEM | **65.4** / 7.15 | 78.6 / **5.65** | 227 / 125 | STR 2.9× 느림 |
| cant | 3D FEM | **59.4** / 4.29 | 63.2 / **3.02** | 150 / 17.7 | STR 2.4× 느림 |
| bmwcra_1 | 3D 구조 | (arena 캡) | 520 / 9.95 | 617 / 49 | STR 1.2× 느림 |
| Transport | 3D (논문 Table-2) | (arena 캡) | 26083 / 0.10 | 23416 / 1.18 | STR **0.90× 빠름** |

- custom vs cuDSS: scircuit·parabolic·cant 모두 **factor 에서 custom 우위**(1.1–1.2×), solve 는 cuDSS 근소 우위.
- front 가 커질수록 STR/cuDSS 비가 **15.6→2.9→2.4→1.2→0.90×** 단조 이동 — 큰 3D 에선 STRUMPACK 이 cuDSS 를 이김.

## 4c. STRUMPACK 논문 재현 검증 (baseline 공정성)

우리 STRUMPACK harness 가 sandbagging 인지 확인하기 위해 논문(Claus·Ghysels·Boukaram·Li, IJHPCA 2025)
Table-2 행렬 **Transport** 에서 우리 harness 를 직접 측정:

| | STRUMPACK factor | cuDSS factor | STR/cuDSS |
|---|---|---|---|
| 논문 (A100) | 3.2 s | 8.8 s | STR **2.75× 빠름** |
| 기존 3090 재현(v8.0, [§04-01](../04-benchmarks-profiling/01-strumpack-paper-reproduction.md)) | 20.4 s | 23.0 s | STR 1.13× 빠름 |
| **우리 harness (3090)** | **23.4 s** | **26.1 s** | STR **1.11× 빠름** |

→ **우리 harness 가 논문을 재현한다**: Transport(3D)에서 STRUMPACK 이 cuDSS 보다 빠르고(1.11×), 기존 3090
재현(1.13×)과 거의 일치, 논문 A100(2.75×)과 방향 일치(3090 FP64=1/64 라 배율만 축소). residual 5e-15(full pivot).
**즉 STRUMPACK baseline 은 공정**하다 — 설계 타깃(큰 3D)에선 논문대로 빠르고, tiny-front(scircuit 15×)에서 느린
건 진짜 약점이지 측정 오류가 아니다. custom 이 tiny-front 에서 STRUMPACK 을 압도하는 결과는 정당하다.

## 5. 한계 / 남은 일

- **G3_circuit / Transport / bmwcra_1**: custom analyze 의 8GB front-arena 상한(`total > 1G doubles`)에 걸림 —
  all-fronts-GPU-resident 설계의 메모리 스케일링 한계(기존 §04-01 에도 기록된 동작). 큰 3D/2D end-to-end 는
  front 스트리밍 재설계 필요. (custom 은 power-grid 및 작은 tiny-front 타깃.)
- **batched 큰-front FP32/TF32 정밀도**: parabolic 같은 stiff 2D-FEM 에선 FP32/TF32 가 부정확(조건수) — FP64 가
  정답 모드. FP32/TF32 는 well-conditioned power-flow(cuPF) 타깃이므로 무관.
- **cuDSS UBATCH 미측정**: *배치 대 배치* 공정 비교는 cuDSS UBATCH 하니스 확장 후 가능.

## 재현
```bash
# baseline
cd exp/harness
LD_LIBRARY_PATH=/opt/magma/lib ./strumpack_bench_magma ../cases_ss/<name>/<name>.mtx 1 3 --sp_reordering_method metis --sp_verbose
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcudss/12 ./cudss_bench ../cases_ss/<name>/<name>.mtx 10
# ours (대칭은 sym_to_full.py 로 full 확장 필요)
python3 ../sym_to_full.py ../cases_ss/parabolic_fem/parabolic_fem.mtx ../cases_ss_full/parabolic_fem
../../custom_linear_solver/build/custom_linear_solver_run --matrix ../cases_ss_full/parabolic_fem/J.mtx \
  --rhs ../cases_ss_full/parabolic_fem/F.mtx --repeat 3 --warmup 1 --single-precision fp64 --serial-nd
```
