# factorize 병목 — ncu 레벨·티어별 분석 (B=1 / 16 / 64)

> **상태**: reference   **갱신**: 2026-06-11
> **한 줄**: factor_small / mid / big 세 커널을 소거트리 레벨별로 ncu 프로파일해, B=1은 latency/under-fill, 배치(B≥16)는 메모리 대역폭이 병목임을 보이고 — occupancy(1 block/SM)와 starved TC는 batch 무관하게 공통임을 확인한다.

대상: case_ACTIVSg70k, tf32(Ozaki), serial-ND 1588, 클럭 고정. B=1은 EXP 직접-issue 경로(`--exp-level-time`, graph 우회), B16/64는 graph 경로를 ncu가 직접 프로파일. launch 순서 = 레벨 bottom-up(잎→root); grid.x = 그 레벨 front 수.

## ncu 메트릭이 실제로 뭔가

표의 약칭이 ncu에서 가리키는 정식 metric 과 의미:

| 약칭 | ncu metric | 의미 |
|---|---|---|
| **warp%** | `sm__warps_active.avg.pct_of_peak_sustained_active` | **achieved occupancy** — SM의 warp 슬롯이 얼마나 찼나(상주 active warp / 최대 48 warp, active cycle 평균). 낮으면 latency 은닉 능력↓. |
| **TC%** | `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | **텐서코어(HMMA) 파이프 가동률** — tensor 명령이 도는 cycle / peak. 100%면 TC가 병목(=바람직). |
| **DRAM%** | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | **device memory(VRAM) 대역폭 사용률** — HBM/GDDR read+write / peak BW. 100%면 메모리 대역폭 bound. |
| **L1%** | `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` | **L1/TEX 유닛 트래픽 강도** — L1 캐시 + **shared memory** 접근이 같은 유닛이라 staging 트래픽이 여기 잡힘. |
| **L2%** | `lts__throughput.avg.pct_of_peak_sustained_elapsed` | **L2 캐시(LTS = L2 slice) 트래픽 강도** — global↔L1 사이 L2 트래픽. |

해석 원리(roofline 식): **어느 자원이 100%에 가까우면 그게 병목**. 전부 낮은데 warp%도 낮으면 → 아무 자원도 안 쓰고 **latency에 묶인 것**(상주 warp 부족으로 stall 은닉 실패).

## 측정 — big tier (FLOP 28%, factorize 시간의 ~60%, 병목 핵심)

대표 레벨(잎쪽 grid 큰 것 / root쪽 grid 작은 것):

| B | grid | warp% | TC% | DRAM% | L1% | L2% | 성격 |
|---|---|---:|---:|---:|---:|---:|---|
| **1** | 21 (L15) | 25 | 2.5 | **3.7** | 6 | 6 | **latency bound** (다 놀음) |
| 1 | 3 (L26) | 30 | 6.9 | 1.5 | 2 | 2 | + under-fill (3 SM) |
| 1 | 2 (상단) | 26 | 4.0 | 0.7 | 1 | 1 | + under-fill (2 SM) |
| **16** | (13,16) | 24 | 2.4 | 20 | 28 | 25 | 메모리 떠오름 |
| 16 | (2,16) | 27 | 5.3 | 15 | 18 | 17 | |
| **64** | (13,64) | 24 | 2.4 | 26 | **35** | **33** | **메모리 bound** (L1/L2 바쁨) |
| 64 | (7,64) | 24 | 2.8 | 32 | **41** | **38** | |

## 측정 — small (FLOP 8.9%) / mid (FLOP 63%)

**small** (TC 원리상 0):

| B | grid | warp% | DRAM% | L1% | L2% |
|---|---|---:|---:|---:|---:|
| 1 | 2246 (잎) | 69 | 42 | 32 | 19 |
| 16 | 20014 | 69 | 49 | 39 | 23 |
| 64 | 80056 | 69 | **50** | 40 | 24 |
| 64 | 15864 | 48 | **67** | 39 | 24 |

**mid**:

| B | grid | warp% | TC% | DRAM% | L1% | L2% |
|---|---|---:|---:|---:|---:|---:|
| 1 | 3542 | 29 | 0 | 17 | 15 | 6 |
| 1 | 1 (상단) | 33 | 6.0 | <1 | <1 | 1 |
| 16 | (56,16) | 30 | 4.2 | 53 | 32 | 16 |
| 64 | (83,64) | 32 | 5.3 | **70** | 44 | 21 |

## 분석 — 병목이 batch 에 따라 바뀐다

1. **occupancy(warp%)는 B=1·16·64 모두 23–33%**(big/mid) — **1 block/SM 한계는 batch 무관**(레지스터 89/116 + shared). small 만 잎에서 48–69%(작은 front 여러 개가 한 블록).

2. **B=1 = latency / under-fill bound**:
   - big: DRAM 0.7–3.7%, L1/L2 1–6%, TC 5% → **컴퓨트도 메모리도 다 놀고** warp가 barrier/load 에서 멈춤(상주 warp 부족으로 못 숨김).
   - 상단 레벨은 grid 1–3 → SM 2–3개만 사용(나머지 ~80 유휴).

3. **B≥16 = 메모리 대역폭 bound**:
   - small·mid 하위: **DRAM 50–70%** (대역폭 거의 포화).
   - big: **L1/L2 35–41% + DRAM 26–35%** — staging(L/U global→L2→L1→shared) + front 데이터 트래픽이 batch 배수로 커져 메모리 계층이 바쁨.
   - B=16은 그 중간(big DRAM 15–26%, L1 18–32%).

4. **TC 는 모든 batch 에서 starved(big ≤7%)** — 원인만 다름: B=1 latency, B≥16 메모리 bound + 낮은 occupancy 로 메모리 latency 못 숨김.

## 레벨별 통합 — tier · occupancy · TC · time% (B=1, 70K)

`s/m/b` = 그 레벨의 small/mid/big front 수. `grid` = 디스패치된 커널의 블록 수(B=1 이므로 = 처리 front 수 = 사용 SM 수). **`time%`** = 그 레벨이 **전체 factorize 벽시계 시간에서 차지하는 비중**(EXP_260611 이 단일스트림 walk 에서 레벨마다 CUDA event 로 측정, 11회 평균).

| level | s/m/b | tier | grid(SM) | warp% | TC% | **time%** |
|---|---|---|---:|---:|---:|---:|
| L0 | 35936/0/0 | small | 2246 | **68.5** | 0 | 2.1 |
| L1 | 13089/0/0 | small | 1637 | **69.8** | 0 | 1.5 |
| L2 | 6578/0/0 | small | 823 | 48.1 | 0 | 1.2 |
| L3–10 | (small 지배) | small | 수백~수천 | ~48–70 | 0 | ~1.3–2.2 ea |
| L11–14 | mid | mid | 27–61 | ~33 | 3–4 | ~1.4–2.2 ea |
| **L15** | 0/19/2 | big(전체21) | 21 | 24.8 | 2.4 | 3.3 |
| L17 | 0/12/1 | big | 13 | 24.7 | 3.1 | 3.4 |
| L19 | 0/6/2 | big | 8 | 25.8 | 4.0 | 3.3 |
| L22 | 0/3/3 | big | 6 | 27.1 | 4.8 | 3.5 |
| **L25** | 0/1/2 | big | 3 | 26.0 | 5.7 | **5.6** ← uc261 spine(최대) |
| **L26** | 0/0/3 | big | 3 | **30.2** | **6.9**← 최고 TC | 4.2 |
| L30 | 0/1/1 | big | 2 | 26.4 | 5.1 | 3.5 |
| L33 | 0/1/1 | big | 2 | 25.4 | 3.7 | 2.8 |
| L34–43 | mid | mid | 1–2 | ~33 | 1–6 | ~0.3–1.9 ea |

**시간 분포 (레벨대별 합)**:

| 레벨대 | occupancy | **time%** |
|---|---|---:|
| L0–10 (small 잎) | **48–70%** | 17% |
| L11–14 · L34–43 (mid) | 24–33% | 18% |
| **L15–33 (big region)** | **24–30%** | **64%** |

관찰:
- **factorize 시간의 64% 가 big region(L15–33)** 에 있고 occupancy 24–30%(1 block/SM, 전 레벨 평탄). **잘 도는 small 잎(occ 48–70%)은 17% 뿐.**
- **small 은 warp-packed** 라 한 블록에 작은 front 여러 개를 채워 occupancy 가 높다(48–70%) — 단 TC 는 원리상 0, DRAM bound(앞 표). (small 커널은 셀 단위라 grid 가 front 수보다 작음.)
- TC% 는 **순수 big 레벨(L26–27)에서 최고 6.9%**, **mid 가 big 커널에 패딩된 하위(L15)에서 최저 2.4%**.
- **최대 시간 레벨 = L25(5.6%)** — uc=261 spine. cap 완화로 TC(5.7%) 타지만 occupancy 26%·grid 3(3 SM) 이라 여전히 느림.

---

## front 수 → 병렬성 → batch → memory bound (B=1 vs B=64, 70K)

상위 레벨일수록 front 가 적어 B=1 에선 GPU 를 못 채우고, batch 가 그 빈 SM 을 채우면 메모리가 병목이 되는지 검증.

**B=1 — big region (grid = front 수 = 사용 SM 수)**:

| level | s/m/b | grid(SM) | warp% | TC% | DRAM% |
|---|---|---:|---:|---:|---:|
| L15 | 0/19/2 | 21 | 25 | 3 | 4 |
| L19 | 0/6/2 | 8 | 26 | 4 | 2 |
| L22 | 0/3/3 | 6 | 27 | 5 | 2 |
| L26 | 0/0/3 | 3 | 30 | 7 | 1 |
| L30–33 | 0/1/1 | **2** | 26 | 4–5 | **1** |

→ 상위 레벨은 front 2–3개 → **82 SM 중 2–3개만** 사용, DRAM 1% (GPU 가 놀고 있음, under-fill).

**B=64 — 같은 tier 측정 범위** (grid = front×64, GPU 꽉 참):

| tier | warp% | TC% | DRAM% | L1% |
|---|---:|---:|---:|---:|
| big | 23–27 | 3 | 22–35 | **27–41** |
| small 잎 | 48–69 | 0 | **50–67** | 38–43 |
| mid 하위 | 24–33 | 4–5 | **60–70** | 33–44 |

### 검증 — 주장은 대체로 맞고, 두 가지 교정

- ✅ **상위 레벨 front 부족 → 병렬성 낮음**: 맞음 (L25–33 front 2–3개 → SM 2–3개).
- ✅ **batch scaling → memory bound**: 대체로 맞음 (B=64 에서 DRAM 이 small/mid 50–70%, big 22–35% 로 급등).
- **교정 1 — "병렬성 낮음"은 두 종류**: B=1 상위 레벨의 손실은 **under-fill**(SM 2–3개만 사용)이 주범. 그런데 **per-SM occupancy 도 23–33%(1 block/SM)** 로 낮은데, 이건 레지스터/shared 한계라 **batch 로 안 풀린다**(B=64 에서도 warp 23%). batch 는 under-fill(SM 채움)만 고친다.
- **교정 2 — "memory bound" 강도가 레벨별로 다름**: **하위(small/mid)는 B=64 에서 DRAM 50–70% = 명확히 대역폭 bound**. **상위(big)는 L1/L2 27–41% + DRAM 25–35% — 메모리 무겁지만 100% 포화는 아니고 occupancy 23% 로 co-limited**(메모리·occupancy 동시 제약).

> **상위 레벨일수록 front 가 적어 B=1 엔 SM 을 거의 못 채운다(under-fill, DRAM 1%). batch 가 빈 SM 을 채워 병렬성을 공급하지만, 그 순간 메모리가 병목이 된다 — 하위는 DRAM 50–70% 대역폭 bound, 상위 big 은 L1/L2 27–41% + occupancy 23% 동시 제약. per-SM occupancy(1 block/SM)는 batch 로도 안 풀린다.**

---

## TC GEMM(trailing) 시간 세분화 (factor_big, B=1, 70K)

trailing 을 `EXP_260611_NO_TRAILING`(전체 스킵) / `EXP_260611_TRAIL_STAGE_ONLY`(staging 후 return) 빌드의 factor_big GPU 시간 차분으로 분할(nsys):

```
factor_big 커널 전체     = 16.48 ms (11 pass 합)
├ non-trailing            = 8.67 ms (53%)   panel-LU + U-solve + extend-add
└ trailing (TC GEMM)      = 7.81 ms (47%)
   ├ staging (L/U→shared) = 0.63 ms (trailing의 8%)
   └ MMA+drain 루프       = 7.18 ms (trailing의 92%)
```

- **non-trailing 이 더 큰 절반(53%)** — TC 와 무관한 panel/solve/extend.
- trailing 안에서 **staging 은 8% 뿐, MMA+drain 루프가 92%**.
- **주의**: MMA+drain 92% 는 *MMA 가 바쁘다*는 뜻이 아니다 — 그 구간 TC 가동은 ~5%(위 표). 즉 **7.18 ms 의 ~95% 는 stall**(barrier/latency, 1 block/SM 라 못 숨김), 실제 MMA 연산은 극소. MMA 와 drain 을 wall-time 차분으로 더 못 나누는 이유도 이것(DCE + stall 지배).
- staging 이 작은 건 **B=1 이 latency-bound**(DRAM 3%)이기 때문. **B=64(메모리-bound)면 staging 비중이 커진다**(단 B=64 는 graph 라 nsys 커널 분리 불가, 이 분할은 B=1 한정).

---

## 결론

| regime | 병목 | 왜 |
|---|---|---|
| **B=1** (단일) | **latency / under-fill** | mid/big 이 1–21 SM 만 쓰고(상단 1–3), 멈춰서 GPU 컴퓨트·메모리 다 놀림 |
| **B≥16** (배치) | **메모리 대역폭** | GPU 꽉 참, 하지만 staging/global 트래픽이 DRAM 50–70%·L1/L2 35–41% 포화 |

→ **두 regime 모두 compute-bound 가 아니다** — 그래서 [`small-tier-no-tensorcore.md`](small-tier-no-tensorcore.md) §5 의 TC cap 완화가 B=64 에서 ~1.02× 밖에 안 된 것이 설명된다(메모리 bound 라 TC 를 더 태워도 무의미).
→ 공통 분모는 **occupancy 23–33%(1 block/SM)**. 이걸 올리면(B≥16) 메모리 latency 은닉이 가능하나, **레지스터 다이어트(`__launch_bounds__(512,2)` 로 89/116→64)는 측정상 net-negative** — 남는 레지스터가 local memory 로 spill 돼 occupancy 이득을 초과(big 케이스 B=256 0.88–0.90×). TF32 Ozaki MMA 커널이 본질적으로 register-bound 이기 때문. staging 트래픽 자체를 줄이는 게 B≥16 의 유일한 직접 레버.
