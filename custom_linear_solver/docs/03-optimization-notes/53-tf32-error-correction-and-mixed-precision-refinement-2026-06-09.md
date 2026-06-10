# TF32 오차보정 + 혼합정밀 정제 — 정확도 회복 연구 통합 2026-06-09

**작성일**: 2026-06-10 (2026-06-09 작업의 통합 정리)
**범위**: 저정밀(tf32/fp16) factor 의 정확도를 FP32~FP64 수준으로 회복하는 두 갈래
연구 — (A) trailing GEMM **내부** 오차보정(TF32x3/x5), (B) solve **바깥** 반복 정제
(iterative refinement / GMRES, 저정밀 factor 를 preconditioner 로). 부수로 발견한 정확도
프로브 스캐폴딩과 직교 perf 실험(fp16 mid TC, fused trail+extend)의 위치/상태를 기록.

원본 작업은 별도 스크래치 사본(`/home/claude/cx-proto`, 삭제됨)에서 진행했고 코드는 아래
표의 커밋들에 보존돼 있다. 본 노트는 그 findings 를 master 에 통합한 것.

---

## 0. 동기 — 저정밀의 relres floor

`docs/27`(USA fp32 conditioning floor) + `docs/28`(thin-K TC ceiling) 이후, **정확도** 축의
질문: tf32/fp16 trailing 의 round-off 를 어디서 회복할 것인가.

- tf32 batch_relres: `ACTIVSg70k` ~0.062, `SyntheticUSA` ~1e-2 (fp64 는 ~1e-12).
- 두 원인이 섞여 있다: **(i) TF32 자체의 10-bit 가수 round-off** (보정 가능), **(ii) 데이터셋
  conditioning floor** — USA Jacobian 은 fp32 로는 cuDSS 레퍼런스도 ~1e-3 floor (`docs/27`).
- → (i) 은 GEMM 내부 오차보정으로, (ii) 는 바깥 반복(고정밀 잔차)으로만 풀린다. 두 레버는
  **상보적**이다.

---

## 1. 레버 A — TF32x3 / TF32x5 error-corrected trailing (채택, opt-in)

**커밋**: `100daa6` (master HEAD), `CLS_TF32X3` CMake 옵션 (default **OFF**).
**위치**: `src/factorize/phases.cuh` `trailing_update_mma_tf32_ptx` 의 `UCP>64` (big) 경로,
`factor_big_tf32_ptx` (`kernels.cuh`).

### 기법 (Ootomo–Yokota / Markidis 계열)
1. **hi/lo 분할**: `tf32t(x)` = 상위 10-bit 마스크(`0xFFFFE000`)로 hi 추출, `lo = x - hi`.
   hi+lo == x 가 **정확** (mma 내부 f32→tf32 반올림과 무관하게 hi 를 그대로 먹임).
2. **x3 → x5 발전**: 단일 tf32 lo 는 10-bit 만 보존 → ill-conditioned front 에서 버린 3-bit 가
   catastrophic cancellation 으로 증폭(~3e-4 floor). 그래서 **lo 를 한 번 더 분할**
   (`lo = lo_hi + lo_lo`)해 13-bit 잔차 전체를 복원 → 5개 mma 항:
   `hi·hi`, `hi·lo_hi`, `lo_hi·hi`, `hi·lo_lo`, `lo_lo·hi` (`lo·lo` 는 버림).
3. **결정적 인사이트 — TC 바깥 누적**: 모든 mma 곱을 **zero-input mma → FP32 SIMT add** 로
   누적. 텐서코어의 누산은 round-toward-zero(RZ)라 작은 보정항을 편향시켜 날려버린다;
   SIMT `+=` 는 round-to-nearest(RN). 이 한 줄이 정확도의 핵심.
4. 레지스터 압력 ↑ → `factor_big_tf32_ptx` 는 x3 시 `__launch_bounds__(512, 1)`
   (아니면 512,2).
5. `CLS_X3_CHECK` 디버그 printf: front0 C[0][0] 의 x3 vs fp32-ref relerr 출력.

### 측정 (sm_86)
- **standalone GEMM 기법 자체는 FP32 등가로 검증** (~2e-7).
- **in-solver**: `ACTIVSg70k` tf32 batch_relres **0.062 → 0.052** (fp32 쪽으로 개선).
- **`SyntheticUSA` 에는 무효** (오히려 약간 악화) — §0 의 conditioning floor 라 GEMM round-off
  보정으로는 못 넘는다. 정확도 한계가 TF32 가 아니라 데이터셋에 있음을 재확인.
- OFF 빌드는 trailing 경로가 기존 단일 mma 와 **byte-identical** (`#else` 분기 = 원본).

### 판단
기법은 유효하나 실익이 **케이스 의존**(70K win, USA 무효) → **default behavior 변경이 아니라
off-by-default 옵션**으로 출하. 정확도가 진짜 필요한 케이스는 레버 B(반복 정제)가 더 보편적.

### scalar 정확도 프로브 (측정용 스캐폴딩, 미커밋)
`trailing_update_staged` 안에 `CLS_TF32X3_SIM`(3-항 scalar 시뮬) / `CLS_TF32X1_SIM`(plain
1-pass tf32) 분기. PTX mma 없이 scalar 경로에서 x3 의 정확도 기여만 격리 측정하는 용도.
`CLS_FP16_CB_SIM`(`cb_store`) 은 부모로 scatter 하는 contribution-block 값을 fp16 으로
반올림해 **fp16-stored CB 의 정확도**를 arena 레이아웃 변경 없이 프로브(메모리 반감은 implied).
모두 일회성 측정 스캐폴딩이라 커밋하지 않음 — 재현 시 본 절 참조.

---

## 2. 레버 B — 바깥 반복 정제 (IR / GMRES, runner 하니스)

**커밋**: `--ir` = `a4fcbe5` (이후 `31a463e` 로 stage1 정리 시 제거, 본 사본이 재도입),
`--gmres` = `142d486` (`exp: lightweight solver experiment code`, `fcab01e` 로 정리).
**위치**: `tests/run_custom_solver.cu` 의 batched-solve 후처리.

저정밀 factor 를 **preconditioner M ≈ A⁻¹** 로 쓰고, 바깥에서 **FP64 잔차**로 정확도를 회복한다.
factor 는 한 번, solve() 만 반복 호출(graph replay).

- **`--ir <steps>`** — mixed-precision iterative refinement. x 는 FP64; 매 스텝 `r = b − A·x`
  (FP64 SpMV) → `d = M⁻¹ r` (저정밀 solve()) → `x += d`. relres < 1e-13 또는 step 소진까지.
  tf32/fp16 factor 로부터 FP 정확도 회복.
- **`--gmres <m>`** — right-preconditioned GMRES(m). M = 저정밀 factor. **참 잔차**
  `‖b − A·x‖` 를 FP64 로 최소화 → 몇 회 반복으로 FP 정확도 도달, IR 보다 ill-conditioned 에
  강함. 매 Arnoldi step 이 `solve()` 1회.

### 판단
USA 처럼 §0(ii) conditioning floor 에 걸린 케이스의 **올바른 해법은 레버 B** — 레버 A 는
GEMM round-off 만, B 는 전체 factor 오차를 잔차로 흡수. 둘은 stackable(저정밀+TC trailing 으로
빠른 factor → 바깥 IR/GMRES 로 정확도). 정량 in-solver sweep 은 이 스크래치 워크트리에
기록되지 않음 — 메커니즘과 하니스만 보존. 후속: 케이스×정밀도별 IR-step / GMRES-m vs wall 표.

---

## 3. 직교 perf 실험 (정확도와 별개, 같은 사본에서 함께 발견)

정확도 작업과 무관하지만 같은 스크래치 사본에 섞여 있던 perf 실험들. 상태/위치만 기록.

| 실험 | 토글 | 상태 | 위치 |
|---|---|---|---|
| FP16 PTX **mid** TC 커널 (`factor_mid_fp16_ptx`) | `CLS_MID_FP16_TC` | 이력에 존재했다 제거됨 ("mid 는 전 정밀도 scalar", latency-bound) | 추가 `74fd837` / 제거 `3f5b959` |
| **fused trailing+extend** (mma 결과를 부모로 직접 atomicAdd, C-drain 왕복 제거) | `CLS_FUSE_{FP16,FP32,TF32}_TRAIL_EXTEND` | fp16 = 채택(`a84d4b7`); tf32/fp32 = **프로토타입, big-front correctness 버그 미해결** | 프로토타입 + 분석 `7cb9d22` (`research/concurrent-fronts`, **doc 30-concurrency-and-c-drain**) |

**fused trail+extend 의 핵심 수확**(상세 doc 30, concurrent-fronts 브랜치): `factor_big_tf32_ptx`
ncu(70K B=64) 에서 **C-drain global STORE 9.98 sector/req (2.5× uncoalesced) + occupancy 33%,
DRAM 8%(latency-bound)**. trailing 비효율은 mma 가 아니라 **uncoalesced C-drain store**.
fused 방향은 USA B=1 −11% 로 유망하나 큰 uc front 에서 correctness 가 깨져 보류. master 미반영.

---

## 4. 보존 위치 요약 (사본 삭제 후 복구 지점)

| 산출물 | 커밋 | 브랜치 |
|---|---|---|
| TF32x3/x5 error-corrected trailing (`CLS_TF32X3`) | `100daa6` | master |
| `--ir` iterative-refinement 하니스 | `a4fcbe5` (제거 `31a463e`) | 이력 |
| `--gmres` preconditioned GMRES 하니스 | `142d486` (정리 `fcab01e`) | 이력 |
| fp16 trailing+extend fusion | `a84d4b7` | master |
| tf32 fused trail+extend 프로토타입 + C-drain 분석 | `7cb9d22` | `research/concurrent-fronts` |
| `factor_mid_fp16_ptx` (mid fp16 TC) | `74fd837` (제거 `3f5b959`) | 이력 |
| scalar 정확도 프로브 (`*_SIM`, `FP16_CB_SIM`) | 미커밋 (본 노트 §1 에 기법 기록) | — |

---

## 5. 결론

- **정확도 회복은 2-레이어**: (A) GEMM 내부 TF32x3/x5 — TF32 round-off 만, 케이스 의존, opt-in
  출하. (B) 바깥 IR/GMRES — 전체 factor 오차를 FP64 잔차로 흡수, ill-conditioned 에 보편적.
- **USA 의 부정확은 TF32 가 아니라 데이터셋 conditioning floor** (`docs/27`) — 레버 A 로 못 넘고
  레버 B(또는 fp64/mixed factor)가 정답.
- TF32x3 의 영구 학습: **보정항은 반드시 TC 바깥(SIMT RN)에서 누적** — TC 의 RZ 누산이 작은
  hi·lo/lo·hi 항을 편향 제거한다.
- 직교 발견: trailing 의 진짜 메모리 병목은 **uncoalesced C-drain store(2.5×)** (doc 30), thin-K
  mma 가 아님 — `docs/28` ceiling 의 메모리측 보강.
