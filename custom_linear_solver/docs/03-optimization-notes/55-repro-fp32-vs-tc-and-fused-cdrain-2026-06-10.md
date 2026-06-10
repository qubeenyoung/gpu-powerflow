# fp32 vs TC 재현 + fused C-drain (fp16/tf32) 검증 — 2026-06-10

**상태**: 재현/측정 로그 + 결론. 코드 변경은 `CLS_FUSE_TF32_TRAIL_EXTEND` 토글 1개(기본 OFF, 미커밋)뿐.
**환경**: RTX 3090 (sm_86), CUDA 12.8, Release, multistream ON, tier-split ON. median of 10, warmup 3.
**대표 케이스 (버킷별 1개)**: case1197(1K~3K) / case3012wp(3K~6K) / case8387pegase(6K~10K) /
case_ACTIVSg25k(25K) / case_ACTIVSg70k(70K) / case_SyntheticUSA(USA).

빌드 라벨:
- **default**: opt-in 플래그 0개. `--precision tf32` = V9h+LB big-tier PTX trailing만 (mid scalar, Ozaki 없음).
- **codex-large**: Codex TC 스택(mid+big TF32 TC + first-order Ozaki), low-fill 번들 OFF.
- **codex-lowfill**: codex-large + low-fill 번들(`SMALL_FRONT_MAX_16` + `MID_TF32_LOW_TC(_FORCE_ALL)` +
  `TC_CLOSURE_PANEL_AMALGAMATE`). ≈ note 51 base policy (단 first-order Ozaki, COLUMN_USOLVE 제외).

---

## 1. fp32 vs tf32 — best-vs-best 통제 비교 (가장 정직한 수치)

둘 다 **codex-large 빌드**에서 **각자 최적 panel-cap**(8~32 sweep), parallel ND. factor+solve per system 기준 `tf32/fp32` (>1 = tf32 빠름):

| 케이스 | B=1 | B=16 | B=64 | B=256 |
|---|--:|--:|--:|--:|
| case1197 (1K~3K) | 1.00× | 1.00× | 1.00× | 1.00× |
| case3012wp (3K~6K) | 1.03× | 0.94× | 0.97× | 1.00× |
| case8387pegase (6K~10K) | 0.96× | 1.02× | 1.03× | 1.00× |
| case_ACTIVSg25k (25K) | 1.05× | 1.02× | 1.06× | 1.07× |
| case_ACTIVSg70k (70K) | 1.16× | 1.03× | 1.10× | 1.06× |
| case_SyntheticUSA (USA) | 1.15× | 1.09× | 1.11× | 1.13× |

- **≤10K: 무가속** (0.94~1.03×, 노이즈). tiny-front라 TC underfill (note 28/54).
- **25K: +5~7%**, **70K/USA: +10~16%** (B=1 피크). 어디서도 1.2× 미달.
- tf32 relres는 cap·케이스별로 FP32 band(≤10K ~2e-5, 25K ~2e-4) ~ 1e-3대(USA), Ozaki로 복원됨.

**방법론 주의**: fp32를 default cap에 묶으면 25K가 1.13×까지 부풀어 보인다. 그러나 fp32도 cap 튜닝하면
1.05~1.07×로 내려옴 — **baseline cap inflation을 조심**해야 한다.

## 2. Codex의 "큰 케이스 1.2×" 주장 — 재현 안 됨, note 54 정정과 일치

같은 codex 빌드·같은 cap·`--serial-nd`, 정밀도만 fp32↔tf32 flip한 통제 비교:

| 케이스 | cap | B=1 | B=64 | B=256 |
|---|--:|--:|--:|--:|
| 25K | 31 | 1.18× | 1.12× | 1.14× |
| 70K | 24 | 1.11× | 1.09× | 1.10× |
| USA | 31 | 1.11× | 1.08× | 1.08× |
| 8387 | 30 | 1.00× | 1.10× | 1.07× |

- note 51/52의 25K 1.24~1.29× / USA 1.26~1.30×는 **재현 안 됨.** 실측 ~1.1×.
- **note 54 §3.2의 자기 정정("1.24×는 cap30 artifact, 진짜 ~1.12×")과 일치** — 본 재현이 그 정정을 확증.
- 원인: note 51/52는 fp32 baseline을 같은 큰 cap(31)에 묶어 **fp32를 느리게** 한 비교라 비율이 부풀려짐
  (25K fp32 cap16=0.139 vs cap31=0.152). **8387 최적 cap은 12~16이고 cap30은 fill을 키워 느린 cap**이다
  (cap28~32가 cap12 대비 15~25% 느림).

## 3. low-fill 번들 (<10K) — 가속 없음, note 54 §3.3 정정

각 구성 자기 최적 cap, `ON/OFF` = codex-lowfill / codex-large (>1 = 번들이 빠름):

| 케이스 | B=1 | B=16 | B=64 | B=256 |
|---|--:|--:|--:|--:|
| case1197 | 1.00× | 1.00× | 1.00× | 1.00× |
| case3012wp | 0.95× | 0.94× | 0.91× | 0.92× |
| case8387pegase | 1.03× | 1.01× | 0.99× | 1.02× |

- case1197 무차별, **case3012wp는 번들이 5~9% 느림**, case8387 tie. → **<10K에서 low-fill 번들 net 이득 0.**
- 여기서 나오는 TC 이득(있어야 ~1.05×)은 전부 base TC 스택에서 나오지 low-fill 플래그가 아님.
- note 54 §3.3 "low-fill 작은케이스 −7~10% real"은 **serial-ND + 고정 cap + 특정 seed**의 fragile 조합에서
  나온 것. **각 구성을 공정 cap 튜닝(parallel ND)하면 우위 소멸** — note 54의 "fragile" 판정을 넘어
  net 이득 없음으로 확증.

## 4. solve는 TC 가속 없음 (Amdahl 천장)

solve(전/후진 삼각 대입)는 fp32·tf32 동일 scalar float 커널 → `tf32/fp32` solve = 0.91~1.09× 노이즈.
그리고 solve가 wall에서 차지하는 비중: 8387 B=1 **46%**, 25K B=64 28%, USA B=1 29%, 8387 B=256 36%.
→ factor만 1.2× 빨라져도 solve 40% 영역이면 전체 `1/(0.6/1.2+0.4)≈1.11×`. **factor+solve 1.1× 천장의 구조적 원인.**

## 5. fused trail+extend (C 왕복 제거) — fp16 성공, tf32 시도

big-front trailing 결과를 부모로 직접 atomicAdd → C global store(F에 쓰기) + readback(extend가 다시 읽기)
왕복 제거. note 30/53이 지목한 uncoalesced C-drain store(9.98 sector/req, 2.5×) 공략.

### 5.1 tf32 구현 (`CLS_FUSE_TF32_TRAIL_EXTEND`, kernels.cuh `factor_big_tf32_ptx`, 기본 OFF)
- **정확도: 통과.** relres가 fused ≈ non-fused (전 케이스 같은 자릿수). **note 53이 보류 사유로 든
  "큰 uc front correctness 버그"는 fp16 패턴 미러링 구현에선 재현 안 됨** — 이전 프로토타입의 다른 이슈였던 듯.
- **속도: B=1 큰 케이스만 이득.** USA B=1 **+8%**, 70K B=1 +3.6%; **B≥64 무효~−2.6%**(USA B256 −2.6%).

### 5.2 fp16 (기본 `CLS_FUSE_FP16_TRAIL_EXTEND=ON`) 동일 패턴, 더 강함
fused ON/OFF factor ms/sys 비율:

| 케이스 | B=1 | B=64 | B=256 |
|---|--:|--:|--:|
| 8387 | 1.05× | 0.96× | 1.02× |
| 70K | **1.33×** | 1.04× | 0.96× |
| USA | 1.01× | 1.00× | **0.90×** |

→ **B=1 대박(70K +33%), 배치 무효~−10%(USA B256).** 기본 빌드가 fuse ON이라 **배치에선 오히려 손해.**

### 5.3 왜 배치에서 안 되나
1. **절감 트래픽이 배치에선 overlap으로 숨겨짐**: B 크면 SM 포화 → C store/load 지연이 다른 batch-block에 가려짐.
2. **fused atomic 패턴 손해**: non-fused는 별도 extend 단계의 선형 sweep으로 코얼레스된 atomicAdd. fused는
   mma 누산기 레이아웃(laneC=0,2,4,6)으로 흩어진 atomicAdd를 trailing에 끼움 → 배치에서 atomic 병목 시 손해.

### 5.4 tf32 spill — 구조적, fp16이 성공한 진짜 이유
`factor_big_tf32_ptx` 레지스터(cuobjdump):

| 빌드 | REG | STACK |
|---|--:|--:|
| non-fused | 64 | 0 |
| fused (Ozaki ON) | 64 | **24** |
| fused (Ozaki OFF) | 64 | **16** |
| fused **fp16** | 64 | **0** |

- fused 부기(`Fp`/`pfsz`/`asm_local`/`abase` + atomic 주소)가 32-reg A-reuse 누산기 위에 얹혀
  `__launch_bounds__(512,2)`의 64-reg 상한 초과 → spill. Ozaki OFF여도 16B → **Ozaki가 아니라 fused 자체가 원인.**
- **fp16 fused는 STACK:0** — fp16 mma는 A를 2-reg(half packing)만 써서(tf32는 4-reg) 부기 흡수할 여유가 있음.
  **fp16 fusion이 성공하고 tf32가 안 된 건 correctness가 아니라 register footprint.**
- per-tile hoist로 spill 죽이기 시도 → **실패(STACK:24 유지)**. binding constraint가 per-element 산술이 아니라
  live captured state + 누산기라서. cheap fix 불가(launch_bounds (512,1)는 occupancy 반토막, 누산기 축소는
  A-reuse 훼손). **게다가 spill은 A-reuse(uc≤64) 경로 — USA/70K big front(uc>64, fall-through, spill 없음)의
  배치 회귀 원인이 아님** (회귀 원인은 5.3의 overlap).

## 6. fp16 정확도 회복 (Ozaki) 가능성

fp16과 tf32는 **가수 10비트로 동일** → 같은 Ozaki 분할(hi/lo + 3-pass mma)로 가수 회복 가능. 단 fp16은
**지수 5비트(±65504)**라 분할 성분이 범위를 벗어나면 overflow(inf)/underflow → 전력망 front의 넓은 동적
범위에서 깨짐(final report §5 "값 범위 underflow"의 원인). 제대로 하려면 **per-block 지수 스케일링** 필요
(tf32엔 불필요). 비용 대비 tf32+Ozaki(스케일링 없이 거의 공짜)를 이길 근거 약함 → 미시도.

## 7. fp16 배치 글로벌 절감의 올바른 경로 (미구현)

현재 fp16은 front를 **fp32 arena**에 저장(`is_fp32_front`) → tf32와 **글로벌 바이트 동일**, fused 트릭은
배치에 무효(5.3). **배치(bandwidth-bound, note 18: big tier DRAM ~69% peak)에서 글로벌 비용을 실제로
줄이려면 fp16 *저장*(front/CB half, 바이트 반감)이 필요** — note 53 `CLS_FP16_CB_SIM` 프로브의 방향.
정확도(range) 비용 동반, 미구현.

---

## 8. 결론

1. **TC(tf32) 가속은 큰 케이스에서만 ~1.1× (25K +5~7%, 70K/USA +10~16% B=1).** ≤10K 무가속. **1.2× 미재현.**
2. **Codex 헤드라인 1.2~1.3×는 baseline cap inflation 산물** — note 54의 self-correction(~1.1×)을 확증.
3. **low-fill 번들은 <10K에서 net 이득 0** (case3012wp는 손해). 공정 cap 튜닝 시 우위 소멸.
4. **solve는 TC 무관 + wall의 30~46%** → factor+solve 1.1× 천장의 구조적 원인.
5. **fused C-drain은 fp16/tf32 둘 다 B=1만 이득, 배치 무효~손해.** tf32 fused는 정확하나(note 53 버그 미재현)
   spill(구조적, fp16은 register 여유로 회피). 둘 다 **B=1/underfill 게이트가 맞다.**
6. 진짜 미개척 lever: 배치 bandwidth를 직접 줄이는 **fp16 저장**(half CB/arena) — 정확도 trade-off 동반.

## 9. 코드 상태
- `src/factorize/kernels.cuh`: `factor_big_tf32_ptx`에 `#ifdef CLS_FUSE_TF32_TRAIL_EXTEND` fused 분기 추가
  (+40줄, **기본 OFF**, 기본 동작 byte-identical). CMake 옵션 미등록 — `-DCMAKE_CUDA_FLAGS=-DCLS_FUSE_TF32_TRAIL_EXTEND`로만 활성.
- 그 외 소스 변경 없음. 측정 원자료: `/tmp/repro_*.tsv` (세션 임시, 영구 보존 필요 시 별도 이관).
