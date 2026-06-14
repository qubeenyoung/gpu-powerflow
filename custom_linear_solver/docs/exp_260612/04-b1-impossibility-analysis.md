# B=1 factorize 가속 불가능성 — 이론·실험 종합 분석

> **상태**: 완료   **날짜**: 2026-06-12   **GPU**: RTX 3090 (82 SM)
> **명제**: B=1 fp32 factorize 의 wall time 은 소거트리의 **critical path(span)** 가 정하는 하드 플로어 근처에 이미 도달해 있고, 남은 자유변수는 **ordering** 하나뿐이며 그 가속 상한은 대형 case 에서 **1.2× 미만**이다. 따라서 "B=1 에서 세 case 일괄 1.2×"는 구현 노력이 아니라 **구조적 한계**로 불가능하다.

---

## 0. 기호와 분해

- `T₁` = B=1 factorize wall time.
- 소거트리는 레벨 `L=0..D-1`(leaf→root). 레벨 `L` 은 `cnt_L` 개의 **독립** front 를 가지며, 각 front 크기 `fsz`.
- 레벨 간에는 **데이터 의존**(부모 front 는 자식들의 extend-add 완료 후에만 시작) → 레벨은 직렬화.
- `t_f` = front `f` 한 개를 한 블록(1 SM)이 인수분해하는 시간(≈ `c·fsz³`, dense LU).

레벨은 직렬, 레벨 내부는 병렬이므로
```
T₁  ≈  Σ_{L=0}^{D-1}  time(level L),     time(level L) = ⌈cnt_L / P_L⌉ · max_{f∈L} t_f
```
여기서 `P_L` = 레벨 L 에서 동시에 돌릴 수 있는 블록 수. 이 식이 이후 모든 논증의 출발점이다.

---

## 1. 하드 플로어 = critical path (Brent / work–span)

**정리(work–span 하한).** 일 `W`, span(=critical path 길이) `S` 인 계산을 `p` 프로세서로 실행할 때
```
T(p) ≥ max(W/p, S)   (모든 p 에 대해, 특히 p→∞ 에서도 T ≥ S)
```
**적용.** 소거트리의 span 은 root→leaf 최장 의존 사슬의 `Σ t_f`. multifrontal 에서 이는 **spine(상위 cnt=1 사슬) + 가장 무거운 subtree 를 관통하는 경로**다. span `S` 는 **프로세서 수와 무관** — SM 이 82개든 무한이든 `T₁ ≥ S`.

> **귀결 1.** B=1 의 깊은 직렬 사슬(spine + 최대 subtree 경로)은 **하드웨어로 못 줄이는 절대 플로어**다. 이 플로어를 바꾸는 유일한 길은 사슬 자체(=트리 구조=ordering)를 바꾸는 것뿐. → §6 으로.

---

## 2. 레버 소거 ①: 스케줄링·오버랩 (GPU 에 유휴가 없다)

`T₁` 을 줄이려면 식 (§0)에서 `time(level L)` 들 사이/내부의 **유휴**를 줄여야 한다. 그러나:

**실험(nsys, graph-off, 25K fp32).** factorize 커널 GPU 시간 합 ≈ **0.72 ms**, 측정 wall ≈ **0.71 ms** → **GPU busy ≈ 100%**. 커널 사이 launch/sync 간극이 무시 가능(이미 CUDA graph 로 캡처·replay).

> **귀결 2.** 회수할 유휴 시간이 없다. "더 나은 스케줄/오버랩/launch fusion"은 **존재하지 않는 슬랙**을 겨냥 → 무효. (실험: launch fusion·DAG 재배치는 graph 가 이미 처리.)

---

## 3. 레버 소거 ②: occupancy (deep level 의 병렬성은 그래프 불변량)

식 (§0)의 `P_L`(동시 블록 수)을 키우면 `⌈cnt_L/P_L⌉` 가 준다. 그러나 한 front = 한 블록(자식 extend-add 의존 + shared-resident)이므로
```
P_L  ≤  cnt_L   (레벨 L 의 독립 front 수)
```
**`cnt_L` 은 소거트리의 그래프 불변량** — 스케줄러가 못 바꾼다. 깊은 레벨은 `cnt_L ≪ 82`:

| (25K, fp32) deep level | cnt_L | 동시 블록 | occupancy |
|---|---:|---:|---:|
| L11 | 21 | 21 | 26% |
| L16 | 7 | 7 | 9% |
| (70K) L26 | 2 | 2 | 2.4% |

**실험(ncu).** deep mid 커널 `sm__warps_active 33%`, `grid 5–29 블록`, `dram 1–4%`. **실험(subtree stream 8→16): 0% 개선**(독립 front 가 없어 stream 을 늘려도 채울 일감이 없음).

> **귀결 3.** deep level 의 occupancy 손실은 **B=1 에서 환원 불가능**하다 — 동시에 돌릴 front 가 물리적으로 1–66개뿐. 이를 채우는 유일한 외생 변수는 **B(배치)**: `P_L ≤ cnt_L·B`. B=1 에서는 닫혀 있다.

---

## 4. 레버 소거 ③: per-front 커널 (자기 SM 위에서 이미 compute-bound)

식 (§0)의 `t_f` 를 줄일 수 있나? deep front(fsz≈120, nc≈16)의 phase 별 FLOP:
```
trailing(병렬)  ≈ uc²·nc ≈ 104²·16 ≈ 1.7e5  MAC   (지배)
panel-LU(직렬)  ≈ nc²·fsz/2 ≈ 1.5e4  MAC   (~1/10)
```
지배 phase 가 **병렬 trailing** 이므로 throughput-bound 후보. **실험(thread 512→1024→`CLS_MID_UNDERFILL_THREADS` 스윕): 0%(±0.2%)**. 512 thread(16 warp)가 이미 한 SM 의 fp32 파이프를 포화 — warp 를 더 줘도(latency-hiding 도, work-per-thread 도) 안 빨라짐.

> **귀결 4.** 한 front 는 **자기 한 SM 위에서 이미 효율적**(compute-bound). 낭비는 *느린 커널* 이 아니라 *놀고 있는 81개 SM*(§3). per-front 가속 레버는 없다.

---

## 5. 레버 소거 ④: front 분할(intra-front 병렬 = idle SM 회수 시도)

§3·§4 의 결론: idle SM 을 쓰려면 **한 front 를 여러 SM(블록)으로 쪼개야** 한다(tiling). 그러나 front 는 dense 이고 L/U 패널을 공유하므로, 블록 분할 시 패널을 **global memory 로 재-staging** 해야 한다.

**실험(이전 prototype, note 07).** tiled-trailing: 13K/25K **0.66–0.72× 회귀**, **B=64(GPU full)에서도 0.62–0.68× 회귀**(70K 포함). ncu: 타일마다 L/U 재-staging 으로 **DRAM +28%, launch +57%**. 즉 회귀는 일감 부족이 아니라 **분할 고유의 통신비용**.

> **귀결 5.** idle SM 회수 = front 분할 = 통신비용 > occupancy 이득. **winning regime 이 없다**(B=64 에서도 짐). 문헌(MAGMA-native, STRUMPACK, Karsavuran'24)도 B=1 작은-root 전용 분할 기법 부재를 확인 — "literature 의 빈 구멍".

---

## 6. 유일한 자유변수 = ordering, 그리고 그 천장

§1–§5 로 스케줄·occupancy·커널·분할이 모두 닫혔다. 식 (§0)에서 남은 건 `{D, cnt_L, fsz}` — **전부 소거트리 구조 = ordering 이 정한다**. ordering 은 `W`(fill=총 일)와 `S`(span=critical path)를 동시에 바꾼다. 따라서:
```
T₁(ordering) ≈ S(ordering) + (병렬영역 일)/P,   둘 다 ordering 의 함수
```
**실험(ordering 민감도, 진짜 fp32 B=1, seed 만):**

| case | parallel-ND median | best-of-many oracle | **ordering 천장** |
|---|---:|---:|---:|
| 13K (8387) | 0.352 | 0.295 (s14) | **1.19×** |
| 25K | 0.744 | 0.707 (s7) | **1.05×** |
| 70K (USA) | 2.202 | 2.038 (s44) | **1.08×** |

- 대형(25K/70K)은 **parallel-ND default 가 이미 oracle 근처**(25K 5% 차) → 줄일 여지가 작다. fp32 는 front 가 작아(바이트 절반) occupancy 가 상대적으로 높고 under-fill 비중이 작기 때문.
- 13K(소형)만 ~1.19× — **노이즈 내 1.2× 경계**이나 초과 못 함.
- best-of-k(`tail_cube` proxy)는 이 천장의 대부분을 결정적으로 회수하지만 천장 자체는 못 넘는다.

> **귀결 6.** ordering 은 듣지만 **상한이 대형 case 에서 1.2× 미만**이고, default 가 이미 그 근처다.

---

## 7. Amdahl 종결 — "완벽 가속" 가정으로도 미달

반례 방어: under-fill 영역을 (불가능하지만) **무한히** 빠르게 만든다고 가정해도?

`f` = under-filled 영역의 시간 비중. 잘 채워진 상위 레벨(수천 small front, occupancy 48–69%)과 spine 직렬 사슬(§1, 하드 플로어)은 **남는다**. 진짜 fp32 에서 상위-filled+spine 비중이 충분히 커서, 부분 가속 상한
```
speedup ≤ 1/(1−f) · (병렬 영역만)  →  대형 case 에서 < 1.2×
```
(§3 의 cost 분해: 25K 는 mid deep L8–23 가 지배하나 그중 cnt 20–66 의 "거의 찬" 레벨이 많아 가속 여지 자체가 작고, 상위 throughput-bound 레벨·spine 이 분모를 키운다.)

> **귀결 7.** under-fill 을 0 으로 만드는 *물리적으로 불가능한* 가정에서도, 잔여 직렬·포화 부분 때문에 대형 case 1.2× 미달.

---

## 8. 결론 (논리 사슬 요약)

```
T₁ ≥ S (span, 하드웨어 무관)                         … §1  [work–span]
   ├ 스케줄 슬랙 없음 (GPU 100% busy)                … §2  [nsys]   → 스케줄 레버 ✗
   ├ deep level 병렬성 ≤ cnt_L (그래프 불변량, ≪82) … §3  [ncu, stream 8→16]  → occupancy 레버 ✗
   ├ per-front 는 자기 SM 위 compute-bound          … §4  [thread 512→1024 = 0%]  → 커널 레버 ✗
   └ front 분할 = 통신비용 > 이득 (B=64 도 회귀)     … §5  [tiling prototype]  → tiling 레버 ✗
⇒ 유일 자유변수 = ordering (W·S 를 정함)             … §6
   └ ordering 천장 = 1.19×(13K) / 1.05×(25K) / 1.08×(70K), default 가 이미 근처
⇒ Amdahl: under-fill→0 가정에서도 대형 <1.2×          … §7
∴ B=1 에서 [13K,25K,70K] 일괄 1.2× 는 구조적 불가.    (cap-inflation 없이는)
```

**탈출구는 단 하나 — B(배치).** §3 의 닫힌 부등식 `P_L ≤ cnt_L·B` 가 B≥16 에서 열리며, 이때만 occupancy·throughput 레버가 부활한다(문헌의 76–253× 가 전부 여기). 이는 실제 power-flow 워크로드(NR·contingency·time-series)와 일치한다.

---

## 부록 — 실험 앵커(재현)

| 논증 | 명령 | 관측 |
|---|---|---|
| §2 busy | `nsys ... --precision fp32`; `cuda_gpu_kern_sum` | Σ kernel 0.72ms ≈ wall 0.71ms |
| §3 occ | `ncu ... factor_mid_blocked<float>` | warps 33%, grid 5–29, dram 1–4% |
| §3 stream | `kMaxSubtreeStreams 8→16` 재빌드 | 0% |
| §4 thread | `CLS_MID_UNDERFILL_THREADS={512,768,1024}` | 0%(±0.2%) |
| §5 tiling | `CLS_TILED_TRAILING=1` (note 07) | 0.62–0.72× 회귀 |
| §6 ordering | `--serial-nd --metis-seed {sweep}` / `CLS_ORDER_K` | 천장 1.05–1.19× |
