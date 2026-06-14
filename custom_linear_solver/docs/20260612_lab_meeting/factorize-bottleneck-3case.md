# factorize 레벨별 병목 — 13K / 25K / 70K, B=1 (ncu)

> **상태**: reference   **갱신**: 2026-06-12
> **한 줄**: case13659(13K)·ACTIVSg25k(25K)·ACTIVSg70k(70K)를 소거트리 레벨별로 ncu 프로파일하니, **13K·25K 는 big tier 가 없는 mid-dominated(시간의 85–87%가 mid), 70K 는 big-dominated(60%)** — 두 regime 으로 갈리지만 **공통적으로 mid·big 전 레벨이 occupancy 1 block/SM(warp 25–33%)·TC starved(<2%)** 라, occupancy 천장은 front 크기·case 와 무관하게 보편적임을 보인다.

기존 [`factorize-bottleneck-ncu.md`](factorize-bottleneck-ncu.md)(70K 단독, serial-ND)를 **3 케이스 + production 기본 ordering** 으로 확장한다.

## 방법

- **빌드**: `CLS_INTERNAL_GRAPH=OFF` (graph 우회). graph-node 프로파일은 꼬리 spine 레벨에서 launch 누락·오귀속이 있어, OFF 빌드의 **non-graph eager launch** 로 ncu 가 커널 1:1 프로파일하게 했다. Release, sm_86.
- **실행**: B=1, `--precision tf32`, `--no-multistream`(단일 스트림 → launch 순서 = 레벨 bottom-up 잎→root), `--warmup 0 --repeat 1`.
- **레벨 매핑**: `--no-tier-split` → 레벨당 정확히 1 launch → launch ID = 레벨(잎→root). 레벨 tier 는 그 레벨 커널(small/mid/big = 레벨 max_fsz 분류)로 라벨.
- **TIER 요약**: 별도 production `tier-split` run 의 launch 를 커널 tier 로 묶어 time-weighted 집계.
- **메트릭**(의미는 [`factorize-bottleneck-ncu.md`](factorize-bottleneck-ncu.md) §"ncu 메트릭" 참조): warp%(`sm__warps_active`), TC%(`sm__pipe_tensor_op_hmma_cycles_active`), DRAM%(`dram__throughput`), L1%(`l1tex__throughput`), L2%(`lts__throughput`), dur(`gpu__time_duration.sum`).
- **ordering**: production 기본(parallel-ND, metis_seed=42). 기존 70K 문서(serial-ND 1588)와 절대 레벨 번호는 다르나 정성 그림은 동일.
- 원자료·스크립트: [`data/level_profile_3case_2026-06-12/`](data/level_profile_3case_2026-06-12/) (`*_notiersplit.csv`, `*_tiersplit.csv`, `report.txt`), [`scripts/level_profile_analyze.py`](scripts/level_profile_analyze.py).

---

## 1. 헤드라인 — 두 regime

**TIER 요약** (production tier-split, B=1, tf32, time-weighted):

| case | factor_total | small time% | mid time% | big time% | 지배 tier |
|---|---:|---:|---:|---:|---|
| **13K** (max fsz≈92) | 0.43 ms | 13.7% | **86.3%** | — (big 없음) | **mid** |
| **25K** (max fsz≈?) | 0.63 ms | 15.1% | **84.9%** | — (big 없음) | **mid** |
| **70K** (max fsz=235) | 1.73 ms | 7.0% | 41.6% | **51.4%** | **big** |

→ **13K·25K 는 big tier 자체가 없다**(max front < 160 → `kFloatSharedFrontMax`). factorize 의 85% 이상이 **mid** 에 있다. 70K 만 big-dominated. 즉 **중형 case 의 병목은 mid, 대형 case 의 병목은 big** — 동일 솔버라도 case 크기로 병목 tier 가 바뀐다.

**tier 별 occupancy / TC (전 case 공통):**

| tier | warp%(occ) | TC% | 성격 |
|---|---:|---:|---|
| small (잎) | 28–51 (잎단 60–69) | 0 | warp-packed, occ 높음, TC 원리상 0 |
| **mid** | **31–33** | **0.3–1.4** | **1 block/SM**, TC starved |
| **big** | **26–27** | **1.5** | **1 block/SM**, TC starved |

→ **mid 도 big 과 똑같이 occupancy 1 block/SM(warp 31–33% / 26–27%)** 이고 TC 거의 0. 기존 문서가 70K big 에서 본 "1 block/SM" 천장이 **mid·전 case 로 일반화**된다.

---

## 2. 레벨별 — occupancy 평탄 천장 (no-tier-split, B=1, tf32)

발췌(전체 [`report.txt`](data/level_profile_3case_2026-06-12/report.txt)). `dur`=레벨 GPU 시간, `time%`=factorize 내 비중.

**13K** (24 레벨, factor_total 0.41 ms) — mid 가 L2–L20, occ 평탄 ~32%:

| L | tier | grid | time% | warp% | TC% | DRAM% |
|---|---|---:|---:|---:|---:|---:|
| 0 | small | 1040 | 4.4% | **66.5** | 0 | 14.0 |
| 1 | small | 258 | 3.0% | 40.0 | 0 | 9.7 |
| 2–20 | mid | 909→1 | ~4–5% ea | **23–33** | 0–1.3 | 5.3→0.4 |
| 21–23 | small | 1 | ~1–4% ea | **2.6–4.3** | 0 | <1 |

**25K** (29 레벨, 0.62 ms) — 동형: 잎 small(occ 42–63%), L4–L26 mid(occ ~33%), 최대 레벨 L16(6.4%, grid 7):

| 레벨대 | tier | time% 합 | occ(warp%) |
|---|---|---:|---:|
| L0–3 (잎) | small | ~9% | 22–63 |
| L4–26 | **mid** | **~87%** | **20–33** (대부분 33) |
| L27–28 (root) | small | ~4% | 2–4 |

**70K** (43 레벨, 1.91 ms) — 잎 small(45–69%) / L4–17 mid(~33%) / **L18–33 big(26–28%) = 시간의 60%**:

| 레벨대 | tier | time% 합 | occ(warp%) | DRAM% |
|---|---|---:|---:|---:|
| L0–3 (잎) | small | ~5% | 43–69 | 35–47 |
| L4–17 | mid | ~25% | 17–33 | 5–24 |
| **L18–33** | **big** | **~60%** | **25–28** | 1–3 (under-fill) |
| L34–42 (root) | mid/small | ~10% | 2–33 | <1 |

→ 세 case 모두 **mid·big 진입 즉시 warp% 가 ~33%(mid)/~27%(big) 로 떨어져 root 까지 평탄**. grid 가 수백(L2 mid grid 909)이어도 33% — **under-fill 이 아니라 per-SM occupancy 한계**(1 block/SM, register+shared). 잎 small 만 warp-packing 으로 40–69%.

---

## 3. 분석 — 병목의 공통 분모와 case별 차이

1. **occupancy 천장(1 block/SM)은 보편**: mid 31–33%, big 26–27%, **case·레벨·grid 무관 평탄**. mid 의 한계는 whole-front shared staging + 128–512 thread 선택, big 은 register(89–116) + L/U staging. 둘 다 batch 로 안 풀리는 per-SM 제약(기존 문서 §교정1 과 일치).

2. **TC starved 보편**: mid ≤1.4%, big ≤2%. compute-bound 아님. B=1 은 latency/under-fill(DRAM 대형 big 에서 1–3%)이라 TC pipe 가 거의 논다.

3. **case 크기 → 병목 tier 이동**:
   - **13K·25K (mid-dominated, big 없음)**: front 가 작아(max≈92) 전부 small+mid. 병목은 **mid 의 1-block/SM occupancy(~33%)**. mid 레벨이 20+개를 root 까지 occ 33% 로 평탄하게 끈다.
   - **70K (big-dominated)**: big 레벨 16개가 시간의 60%, occ 27%, grid 2–10(상단 root 쪽은 under-fill 까지 겹침).

4. **시간 분포의 비대칭**: 잎 small 은 occ 가 제일 높은데(45–69%) 시간은 5–15% 뿐. 시간의 85%(중형)·90%(대형)가 occ 25–33% 인 mid/big 에 있다. **"잘 도는 곳은 일이 적고, 일이 많은 곳은 occ 가 낮다."**

---

## 4. 함의 — 260612 목표로의 연결

- **목표 3(big front tiling)**: 70K 에서 big = 시간 60% · occ 27% 재확인 → 대형 case 최우선 레버 맞음. **단, 13K·25K 엔 big 이 없다** → big tiling 은 **대형 case 전용** 이득. 중형까지 커버하려면 동일 처방(1-block/SM 깨기)을 **mid 에도** 적용해야 한다(mid 도 occ 33% 평탄).
- **목표 2(mid/big 경계)**: 13K·25K 가 순수 mid 라 mid 커널 효율이 그 case 전체를 좌우. mid occ 가 grid 충분(수백)에도 33% 인 건 **whole-front shared staging + thread 수(128–512) 선택** 탓 → 경계뿐 아니라 **mid kernel 의 thread/shared 점유 모델**도 같이 봐야 함. mid occ 를 올리는 게 중형 case 의 직접 레버.
- **공통**: 세 case 모두 **occupancy(1 block/SM)가 단일 천장**. front 를 더 잘게(tiling) 쪼개 barrier/latency 를 은닉하는 방향이 mid·big 양쪽에 유효. TC 를 더 태우는 건(이미 starved) 무의미.

> 다음: 목표 3 을 **big + mid 양 tier 의 occupancy(1-block/SM) 깨기**로 일반화하는 게 13K/25K/70K 를 모두 커버하는 길. 단일-front tiling(trailing 2D 분할) 프로토타입 → ncu warp% 회복 검증 → Amdahl.
