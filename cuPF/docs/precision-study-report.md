# 전력조류 혼합정밀(Mixed Precision) 연구 보고서

**주제.** cuPF의 Mixed 프로파일(선형계는 FP32, 나머지는 FP64)이 왜 정당한가를 "되더라"가 아니라
**수치해석 이론 + 측정**으로 증명하고, 어떤 연산을 FP32로 바꿀 수 있고 없는지를 가른다.

**방법.** 연산별 정밀도를 정확히 제어하는 controlled NR 전력조류 하니스(NumPy/SciPy, 진짜 FP32 희소
LU=`splu`, FP32 저장은 round-trip 캐스팅으로 모사). 케이스는 pypower(case9–300)+pandapower(pegase/
rte/sp/GB), 조건수 κ는 **실제 cuPF NR Jacobian**(`cls_linsys/*.mtx`, ACTIVSg70k까지)로 교차검증.
정확도는 항상 현재 상태에서 **FP64로 재계산한 TRUE ‖F‖∞**로 채점(저정밀 솔버가 "수렴했다"고 착각해도
정직하게 평가). 재현: `cuPF/tests/precision_study/`. 상세 문서: `mixed-precision-plan.md`,
`mixed-precision-findings.md`, `precision-error-measurement.md`.

`u₃₂ = 2⁻²⁴ ≈ 5.96e-8` (FP32 unit roundoff).

---

## 1. 결론 요약 (TL;DR)

1. **Mixed는 구조적으로 옳다.** FP32 선형해는 **inexact Newton의 유효 forcing term**(η<1)이고, 최종
   정확도는 **잔차 정밀도(FP64)**가 정하므로(혼합정밀 반복정련) — Mixed는 **FP64와 동일한 반복수·정확도**
   에 도달한다. (§3)
2. **full-FP32는 ~1e-3에서 바닥**나며, 그 원인은 **V·Ybus를 FP32로 *표현***하는 것이고(합산 오차 아님),
   전력식의 **상쇄(cancellation)로 증폭**된다. 보정합산(Kahan/FP64-누산)으로 **못 깬다.** (§4)
3. **비대칭의 증명:** 선형해는 지배 지표가 **후방오차 ω≈u₃₂**(κ 무관·자기보정)라 FP32 가능, 잔차는
   **합산 조건수 κ_sum~1e2–1e4의 전방오차**(보정 불가·답을 정의)라 FP32 불가. (§5)
4. **근본 원인은 데이터 분포:** 미지수 |V|는 좁아(0.9–1.26) 양호하나, 전력식 `S=V·conj(YV)`이
   물리적 상쇄로 ill-conditioned. (§6)
5. **공학적 함의:** 정밀-임계 연산(state·mismatch)은 **싼 O(n)** 이고, 비싼 factor/solve만 FP32로 안전 →
   **Mixed가 정밀도 배분의 최적점**. "더 FP32로" 미는 것은 이득 없고 정확도만 무너뜨린다. (§7)

---

## 2. 현재 Mixed의 연산별 정밀도 (코드 확인)

`CudaMixedStorage : CudaBatchedStorage<double,float>`.

| 연산 | FP64 | **Mixed** | FP32 |
|---|---|---|---|
| state V/Va/Vm/Sbus | double | **double** | float |
| Ibus = Ybus·V | double 누산 | **double 누산** | float |
| mismatch F, ‖F‖ | double | **double** | float |
| Jacobian 값 | double | **float** | float |
| factorize / solve (dx) | double | **float** | float |
| voltage update | double | **double(상태)** | float |

즉 **Mixed = 잔차체인(Ibus·F·norm)+상태 FP64, Jacobian+선형해만 FP32.**

---

## 3. Part A — Mixed가 옳은 이유 (근거 + 측정)

| case | n | κ₁(J) | FP64 it/‖F‖ | **Mixed it/‖F‖** | FP32 it/‖F‖ | Mixed η̄ |
|---|---|---|---|---|---|---|
| case118 | 118 | 5.7e3 | 4 / 1.1e-10 | **4 / 1.2e-10** | 49 / 1.3e-5 | 9.1e-7 |
| case300 | 300 | 2.4e5 | 5 / 1.4e-12 | **5 / 8.1e-11** | 49 / 3.0e-4 | 3.0e-5 |
| case1354pegase | 1354 | 6.8e5 | 5 / 3.2e-12 | **5 / 1.6e-10** | 49 / 2.0e-3 | 2.5e-5 |
| case3120sp | 3120 | 1.4e6 | 6 / 1.0e-11 | **6 / 1.5e-11** | 49 / 4.1e-3 | 5.4e-5 |
| case9241pegase | 9241 | 1.3e7 | 6 / 4.6e-12 | **6 / 7.5e-10** | 49 / 2.2e-3 | 1.8e-4 |
| GBnetwork | 2224 | 1.9e7 | 5 / 6.7e-11 | **5 / 1.3e-9** | 49 / 2.3e-2 | 3.5e-4 |

**두 고전 이론으로 정당화 (둘 다 전력조류 Jacobian에서 측정 확증):**
- **Inexact Newton (Dembo–Eisenstat–Steihaug 1982).** `J·dx=F`는 `‖F−J·dx‖≤η‖F‖`, η<1만 만족하면
  수렴. FP32 분해의 forcing term η̄=**1e-6 … 4e-4 (항상 <1)**, κ와 함께 커지나 1에 한참 못 미침 →
  **Mixed 반복수 = FP64 반복수.**
- **혼합정밀 반복정련 (Carson–Higham).** NR 루프는 defect correction(FP64 잔차, FP32 step, FP64 갱신)
  이라 **최종 정확도 = 잔차 정밀도(FP64)**, 분해 정밀도(FP32) 아님 → **Mixed가 FP64급 ‖F‖ 도달.**

**보수적 경계와 실제 (정직성 보정).** 최악경계는 FP32 전방오차 ≲ κ·u₃₂라 κ≳10⁷(κ·u₃₂≈1)에서 위험을
경고하나, **실제 cuPF Jacobian의 FP32 단일해 전방오차는 그보다 훨씬 작다:**

| (cls_linsys 실 Jacobian) | 1354 | 9241 | ACTIVSg25k | **ACTIVSg70k** |
|---|---|---|---|---|
| κ·u₃₂ (최악경계) | 0.04 | 0.8 | 1.6 | **42** |
| FP32 solve 전방오차 | 1e-5 | 2.9e-4 | 1.7e-4 | **2.2e-3** |
| FP32 solve η | 4e-6 | 3.3e-6 | 2.0e-4 | **1.1e-3** |

ACTIVSg70k(κ·u₃₂=42)에서도 η=1.1e-3<1 → **FP32 선형해(Mixed)는 70k까지 견고.** 70k가 FP64를 요구하는
통설은 *선형해*가 아니라 *표현 바닥*(§4) 때문이다.

---

## 4. Part B — 다른 연산도 FP32로? 1e-3 바닥의 정체

연산별 ablation (최종 TRUE ‖F‖∞):

| config | case300 | 1354pegase | 3120sp | 9241pegase |
|---|---|---|---|---|
| FP32 (all) | 4.4e-4 | 1.7e-3 | 4.8e-3 | 2.2e-3 |
| FP32, **state만 FP64** | 4.2e-4 | 1.5e-3 | 3.4e-3 | 3.1e-3 |
| FP32, **residual만 FP64** | 1.0e-4 | 1.2e-3 | 1.9e-3 | 1.8e-3 |
| FP32, **residual = FP64누산/FP32저장(보정)** | 1.6e-4 | 2.3e-3 | 3.0e-3 | 4.3e-3 |
| FP32, state+residual(보정) | 1.8e-4 | 6.6e-4 | 2.0e-3 | 1.5e-3 |
| **Mixed (state+residual FP64)** | **8.1e-11** | **1.6e-10** | **1.5e-11** | **7.5e-10** |

**발견 (가설을 부분 반증):** state만/residual만/**보정합산(FP64누산+FP32저장)** 전부 바닥 못 깸. 오직
state+residual 둘 다 FP64(=Mixed)만 1e-10. ⇒ 바닥은 **합산 오차가 아니라 V·Ybus의 FP32 *표현*** 때문.

**입력만 FP32로 반올림(나머지 산술 FP64)해 분해 — 수렴해 V*에서:**

| | 1354pegase | 9241pegase |
|---|---|---|
| ‖F‖ FP64 | 3e-12 | 5e-12 |
| **V만 FP32** | 6.2e-4 | 2.0e-3 |
| **Ybus만 FP32** | 7.3e-4 | 7.8e-4 |

V 또는 Ybus 하나만 FP32여도 1e-12 → ~1e-3. **합산정밀로는 복원 불가**(정보가 합산 *전* 입력 라운딩에서 손실).

---

## 5. 비대칭의 증명 — 과학적 측정 지표

| 연산 | 지배 지표 | FP32에서 값 | 하류 보정? | FP32 가부 |
|---|---|---|---|---|
| Jacobian+factor+solve | **후방오차 ω** (Rigal–Gaches) | ω≈u₃₂ (κ무관) | **예** (매 iter 재계산) | **가능** |
| state V+Ibus+mismatch | **κ_sum 상쇄의 전방오차** | κ_sum·u₃₂≈1e-3 | **아니오** (F가 답) | **불가** |

**증명 A (선형해 후방안정):** 무작위 RHS로 측정한 FP32 LU 후방오차
`ω=‖b−Jx̂‖/(‖J‖‖x̂‖+‖b‖)`:

| case | κ(J) | **fp32 ω** | fp32 전방오차 |
|---|---|---|---|
| case118 | 5.7e3 | **1.8e-8** | 3.0e-6 |
| case1354pegase | 6.3e5 | **1.7e-8** | 1.9e-5 |
| case9241pegase | 1.3e7 | **2.5e-8** | 1.9e-4 |

ω≈u₃₂로 **κ(J) 무관**(κ 5.7e3→1.3e7) ⇒ FP32 LU는 **후방안정**(Higham, ASNA ch.9): 계수행렬을 ~u₃₂
만큼만 바꾼 정확해. NR에선 후방오차(=forcing term η)만 중요하고, 전방오차 κ·u₃₂는 다음 iter가 자기보정 →
**FP32 분해는 수렴해를 못 움직임.**

**증명 B (잔차 ill-conditioned + 보정불가):** running-error 예측 vs 실측 바닥:

| case | u₃₂·median Σ\|Y_ij V_j\| | u₃₂·max Σ\|Y_ij V_j\| | **실측 FP32 바닥** |
|---|---|---|---|
| case118 | 4.8e-6 | 4.6e-5 | **1.6e-5** |
| case1354pegase | 4.9e-5 | 2.1e-3 | **7.6e-4** |
| case9241pegase | 3.3e-5 | 3.1e-3 | **1.6e-3** |

실측 바닥 = **running-error 한계 `|ΔF_i|≈u₃₂·Σ_j|Y_ij V_j|`**(Higham ASNA ch.3–4)와 일치. 상대오차
= κ_sum·u₃₂, κ_sum=Σ|Y_ij V_j|/|Ibus_i| (중앙값 2e2–3e4). 잔차는 후방오차 탈출구도, 자기보정도 없어
(F가 고정점/수렴기준을 정의) 이 바닥이 그대로 정확도를 가둠.

---

## 6. 근본 원인 — 입출력 데이터 분포 (50/90/99/max)

| case | \|V\| (미지수) | \|Y_ij·V_j\| (합산항) | \|Ibus\|=\|S\| (순주입) | κ_sum 중앙값 |
|---|---|---|---|---|
| case118 | 0.99 / 1.02 / 1.05 | 15 / 67 / 187 / **389** | 0.37 / 1.5 / 4.9 / 6 | **217** |
| case1354pegase | 1.04 / 1.07 / 1.08 / 1.11 | 154 / 1.6e3 / 5e3 / **1.7e4** | 0.45 / 2.6 / 12.7 / 32 | **3.0e3** |
| case3120sp | 1.06 / 1.08 / 1.12 / 1.26 | 85 / 1.4e3 / 1.5e4 / **2.5e4** | 0.047 / 0.19 / 3.1 / 18 | **3.4e4** |
| case9241pegase | 1.03 / 1.06 / 1.10 / 1.18 | 74 / 1.5e3 / 5.3e3 / **2.6e4** | 0.21 / 2.1 / 9.5 / 34 | **2.7e3** |

- **|V| 좁음(0.9–1.26):** 미지수는 양호 — FP32로 표현해도 그 자체는 문제없음.
- **합산항 |Y_ij·V_j| 큼(최대 ~1e4):** 버스로 유입되는 선로 전류가 큼.
- **순주입 |Ibus|=|S| 작음(중앙값 0.05–0.5):** 그 큰 흐름들이 버스에서 거의 상쇄(Kirchhoff: 유입≈유출).
- **κ_sum = 큰 합산항 / 작은 순주입 ≈ 1e2–1e4:** 이 상쇄가 u₃₂ 입력 라운딩을 ~1e-3 잔차오차로 증폭.

⇒ FP32가 "일반적으로 거칠어서"가 아니라, **미지수 V는 양호한데 전력식 `S=V·conj(YV)`이 물리적 상쇄로
ill-conditioned**이고, 선형해의 지배 지표(후방오차)는 그 상쇄를 거치지 않기 때문. **같은 u₃₂, 정반대 결과.**

---

## 7. 공학적 함의 및 결론

| 연산 | 비용/iter | 정밀 임계? | FP32 |
|---|---|---|---|
| state V, voltage update | O(n) — 쌈 | **예 (FP64 필수)** | 불가 |
| Ibus, mismatch F, ‖F‖ | O(nnz)≈O(n) — 쌈 | **예 (FP64 필수)** | 불가 |
| Jacobian 조립 | O(nnz) | 아니오 | 가능 |
| **factorize + solve** | **O(비쌈) — 병목** | 아니오 | **가능** |

정밀-임계 연산이 곧 **싼 연산**이라 FP64로 둬도 비용이 거의 없고, **비싼 분해/solve만 FP32로도 안전**.
따라서 **Mixed는 정밀도 배분의 최적점**이며, 1e-3 바닥은 "돌파"할 가치가 없다(돌파하려면 V/Ybus를
FP64 또는 동급 double-single로 둬야 하는데, 그러면 FP32의 의미가 사라지고 비용도 없던 절감이라).

**재사용 가능한 측정 레시피:** 임의 연산에 대해 ① 지배 지표가 후방오차냐 전방오차냐, ② 조건수는
(선형해 κ(J), 리덕션 κ_sum=Σ|항|/|합|), ③ 하류 자기보정 여부 — 이 셋으로 **timing 전에 FP32 가부를
선험적으로 판정**.

---

## 부록 — 산출물 / 미실행 / 출처

- **하니스:** `cuPF/tests/precision_study/` (`harness.py`, `driver_A.py`, `driver_B.py`, `measure.py`,
  `resA.json`, `resB.json`).
- **세부 문서:** `mixed-precision-plan.md`(근거·계획), `mixed-precision-findings.md`(A/B 결과),
  `precision-error-measurement.md`(측정법·증명).
- **미실행(데이터 한계):** ACTIVSg25k/70k의 *end-to-end Mixed NR*은 Ybus/Sbus가 없어(cls_linsys엔
  선형계만) 미실행 — 대신 그 실 Jacobian에서 FP32 *solve*가 견고함(η<1)은 확인. FP32 저장은 캐스팅으로
  모사(혼합정밀 연구 표준 기법); GPU 실행은 timing 확인용(정확도는 정밀도로 결정되어 충실히 재현됨).
  κ는 1-노름 추정(Hager/`onenormest`).
- **이론 출처:** Higham, *Accuracy and Stability of Numerical Algorithms* (후방오차·running-error·
  합산 조건수); Dembo–Eisenstat–Steihaug 1982 (inexact Newton); Carson–Higham 2017/2018, Higham–Mary
  2022 (혼합정밀 반복정련); Wang/Fraunhofer 2021 (이 결과가 반증하는 "FP64 for stability" 가정).
