# 공정 재검증: STRUMPACK 오더링 튜닝 + 우리 NodeNDP 이식 + panel_width 최적화

> **상태**: 실측 완료(25k·USA·3120sp)   **갱신**: 2026-06-17
> **한 줄**: §5d가 STRUMPACK을 `and` 정렬로만 테스트해 **과소평가**했다. 공정하게 NodeNDP+nd_param을
> sweep하면 STRUMPACK은 우리 동작점을 능가하는 (깊이,fill)에 도달한다 → **헤드라인 16–66×는 ~10×로 줄어든다.**
> 그래도 깊이·fill을 양쪽 다 통제한 뒤 **factor ~10–13×, solve ~12–16× 우위가 남고**, 이것이 ordering/깊이/fill로
> 설명 안 되는 순수 커널 효율이다. 부수 발견: **기본 `panel_width=16`은 비최적 — pw=8이 전 케이스에서 우위.**

선행: [§06 head-to-head](06-head-to-head-2026-06-16.md)(특히 §5d·§5e), [main-report §3](../main-report.md).

---

## 0. 동기 — §5d의 confounder를 끝까지 통제

§5d는 "STRUMPACK 트리를 얕게 만드는 유일 옵션은 `and`이고 fill을 6.6~9× 치른다"고 했다. 그러나 STRUMPACK 자신의
경고가 권한 **`NodeNDP`(METIS_NodeNDP)는 테스트하지 않았다.** 또 "우리에겐 nd_param 같은 오더링 손잡이가 없다"는
주장도 검증되지 않았다. 이 둘을 직접 친다.

---

## 1. STRUMPACK 공정 튜닝 — NodeNDP + amalgamation + nd_param sweep

`strumpack_bench_magma`에 `set_from_command_line`로 옵션 주입(재빌드 불요). 25k, FP64, GPU, per-sys factor는
`update_matrix_values`로 강제 재분해. amalgamation(`--sp_enable_MUMPS_SYMQAMD --sp_enable_agg_amalg`)도 켬.

| 구성 | levels | fill | factor/sys ms | solve/sys ms |
|---|---|---|---|---|
| **우리 (pw=16 기본)** | 26 | 451k (2.0×) | **1.24** | **0.63** |
| metis NodeND (STRUMPACK 기본) | 78 | 229k | 24.6 | 23.5 |
| **`and` (§5d가 쓴 것)** | 23 | 1515k (6.6×) | 28.6 | 14.3 |
| NodeNDP nd_param=2 | **14** | **384k (1.67×)** | 12.8 | 9.3 |
| NodeNDP nd_param=4 | 13 | 553k | 16.1 | 6.7 |
| NodeNDP nd_param=8 | 12 | 898k | 14.0 | 6.7 |
| NodeNDP nd_param=16 | 11 | 1539k | 16.6 | 6.6 |

**정정 1 (STRUMPACK에 유리한 정정):** §5d는 STRUMPACK을 과소평가했다. **NodeNDP nd_param=2면 14층 / fill
384k(1.67×)** — *우리 동작점(26층, 2.0×)을 깊이·fill 두 축 모두에서 능가*한다. `and`(23층, 6.6×)는 STRUMPACK
최적이 아니었다. 또 SYMQAMD+agg_amalg는 factor를 16→12ms로 ~1.3× 개선(amalgamation이 실제 작동·측정 가능).

**함의:** "STRUMPACK은 우리 (깊이,fill) 동작점에 도달 못 한다"는 §5d의 주장은 **반증됐다.** ordering/amalgamation은
우리 차별점이 아니다 — STRUMPACK이 (더 정교한 MUMPS QAMD로) 같은 일을 한다.

---

## 2. 우리 솔버에 NodeNDP+npes 이식 — 그리고 그것이 무력한 이유

"노출만 안 한 것 아니냐"를 검증하려고 실제로 붙였다. `metis.h:220`에 `METIS_NodeNDP(...,npes,...)`가 있고
libmetis.so에 export됨(`T METIS_NodeNDP`). `src/analyze/reorder/metis_nd.cpp` serial 경로에 env 토글
**`CLS_ND_NPES`** 추가(off면 기존 NodeND). STRUMPACK의 nd_param이 먹이는 바로 그 `npes`다.

**우리 솔버 npes sweep (25k, FP64, B=1):**

| npes | levels | fill | factor ms | solve ms |
|---|---|---|---|---|
| NodeND (기본) | 23 | 470,539 | 1.28 | 0.64 |
| 4 / 16 / 128 | 23 | 470,539 | 1.29 | 0.64 |
| 256 | 23 | 468,932 | 1.20 | 0.64 |
| 1024 | 23 | 451,334 | 1.21 | 0.60 |
| 4096 | 23 | 481,383 | 1.21 | 0.62 |

(정확성 정상: residual 2.4e-14. npes≥256에서 ordering이 실제 바뀜이 fill 변화로 확인 — 코드는 작동 중.)

**정정 2 (아키텍처적 발견):** **npes는 우리 솔버에서 무력하다** — levels는 npes 4~4096 전 구간 23 고정, fill ±5%.
STRUMPACK에선 nd_param이 levels 78→8을 움직였는데(§1) 우리는 안 움직인다. 이유:

- **STRUMPACK**은 오더링 루틴의 separator tree를 *그대로 front tree로 사용* → npes가 그 트리를 직접 빚음.
- **우리**는 오더링에서 **permutation만 취하고 separator tree(`sizes`)를 버린 뒤, 자체 amalgamation으로 트리를
  재구성** → 깊이/fill 손잡이가 **ordering이 아니라 amalgamation(panel_width)에 산다.** npes를 돌려도 amalgamation이
  23층으로 평탄화한다.

⇒ "nd_param 등가 ordering 손잡이가 우리에게 없다"는 표면적으론 맞지만(stock NodeND 사용), 본질은 **우리 설계에선
ordering 손잡이가 redundant**하다는 것. 깊이/fill 제어는 panel_width가 담당하며, 이는 ordering이 아니라 amalgamation
계열이다(STRUMPACK의 agg_amalg와 같은 family).

---

## 3. 부수 발견 — 기본 `panel_width=16`은 비최적

panel_width(=amalgamation cap, `relaxed_panels`)는 우리 (깊이↔fill) 손잡이다. **단 기본 빌드는 cap을 무시**
(pw≈16 고정); `-DCLS_RESPECT_PANEL_CAP=ON`(`build_pcap`)에서만 작동. cap 존중 빌드로 sweep:

**25k (B=1, FP64), build_pcap:**

| pw | levels | fill | factor ms | solve ms |
|---|---|---|---|---|
| 1 (병합 없음) | 148 | 1,652,666 | 1.58 | 1.48 |
| **8** | 31 | 531,727 | **0.91** | **0.54** |
| 16 (현재 기본) | 23 | 470,539 | 1.28 | 0.64 |
| 32 | 18 | 437,288 | 1.05 | 0.57 |

pw=16이 이웃값(pw=8·32)보다 나쁜 **국소 최악점**(factor 1.28 vs 0.91, 3 trial 재현). STRUMPACK nd_param과 정반대로
**우리는 병합할수록 깊이도 fill도 같이 준다**(pw1의 1.65M = front당 패딩 낭비를 병합이 amortize).

**전 케이스 pw=8 vs 기본 pw=16:**

| 케이스 | pw=8 (f/s ms) | pw=16 (f/s ms) | pw=8 우위 |
|---|---|---|---|
| case3120sp (6k) | 0.290 / 0.189 | 0.304 / 0.207 | factor 1.05×, solve 1.10× (pw=4가 근소 최적) |
| ACTIVSg25k (25k) | 0.91 / 0.54 | 1.28 / 0.64 | **factor 1.41×, solve 1.19×** |
| SyntheticUSA (82k) | 2.28 / 1.28 | 2.37 / 1.35 | factor 1.04×, solve 1.05× |

⇒ **pw=8이 세 케이스 모두에서 기본 pw=16보다 우위**(25k 대폭, 나머지 ~5%). 이건 **표준 relaxed-supernode 튜닝이지
신규성 아님** — 기본값이 비최적이라 성능을 흘리던 것.

**정밀도/체제별 확인 (2026-06-17):**
- **fp32 single (B=1)**: fp64와 동일하게 pw=8 명확 우위 (25k 0.904/0.531 vs pw16 1.277/0.632; USA 2.279/1.284 vs 2.373/1.347).
- **tf32 batched (B=64)**: pw=8과 pw=16 **사실상 동률**(~1%). factor는 pw=8 근소 우위(0.0318 vs 0.0322/sys),
  solve는 pw=16/32 우위(0.0169/0.0158 vs 0.0176). B=64는 배치가 GPU를 채워 작은-front occupancy 이득이 희석되고
  얕은 트리(큰 pw)의 solve launch 절감이 더 먹힘 — B=1과 반대 체제.

⇒ **결론: pw=8 기본값은 B=1(cuPF NR 루프 실제 체제)에서 fp64·fp32 모두 이득, tf32 배치에선 무해(동률).** 안전한 변경.
구현: `lower.cu` `compute_effective_panel_width`의 `(n>=16000)?16:..` 휴리스틱 제거 → config `max_panel_width=8` 직접 사용.

---

## 4. 최종 fair head-to-head — 깊이·fill·ordering 모두 통제

| 구성 | levels | fill | factor ms | solve ms |
|---|---|---|---|---|
| **우리 pw=8** | 31 | 532k | **0.91** | **0.54** |
| **우리 pw=24** | 19 | 433k | **1.22** | **0.57** |
| STRUMPACK NodeNDP nd_param=4 | 13 | 553k | 16.1 | 6.7 |
| STRUMPACK NodeNDP nd_param=2 | 14 | 384k | 12.8 | 9.3 |

비슷한 동작점(13~19층, 380~550k fill)에서: **factor 1.2 vs 12.8~16 → ~10–13×**, **solve 0.57 vs 6.7~9.3 → ~12–16×.**

---

## 5. 결론 (헤드라인 정정)

1. **헤드라인 16–66× → 공정 튜닝 후 ~10–16×.** 원래 격차의 일부는 STRUMPACK을 기본 정렬(78층 deep-tree)로
   돌려서 부풀려졌다. NodeNDP로 공정 튜닝하면 factor 16×→~10–13×, solve 33–66×→~12–16×.
2. **ordering·amalgamation·깊이·fill은 차별점이 아니다.** STRUMPACK도 한다(NodeNDP/SYMQAMD), 일부는 더 잘한다.
   우리 NodeNDP 이식은 아키텍처적으로 무력(separator tree를 안 씀). 깊이/fill은 우리 쪽 amalgamation(pw) 담당.
3. **남는 ~10× 우위는 깊이·fill·ordering 통제 후의 순수 커널/구현 효율**: tiny packing+fusion occupancy
   (2%→30–59%, §06 §5b) + GPU-resident solve vs MAGMA vbatched(occupancy 2%) + STRUMPACK solve 오버헤드.
   단 §06 §5e ablation대로 **B=1에선 packing+fusion 단독 기여는 1.27×**(tiny tier가 factor의 ~43%, Amdahl 천장
   ~1.74×)이고, 잔여는 amalgamation·graph·STRUMPACK 구현 오버헤드의 합 — "커널이 단일 주원인"은 B=64 한정.
4. **부수**: 기본 panel_width=16 → 8로 내리면 power-flow 전 케이스에서 factor/solve 개선(25k 최대 1.41×).

## 6. 재현

- 우리 NodeNDP: `CLS_ND_NPES=<npes>` (serial-nd 경로, `metis_nd.cpp`). 필요시 `CLS_ND_NPES_SWAP`로 perm/iperm 전환.
- 우리 pw sweep: `build_pcap`(`-DCLS_RESPECT_PANEL_CAP=ON`) + `--max-panel-width N`.
- STRUMPACK: `strumpack_bench_magma J.mtx 10 3 --sp_reordering_method metis --sp_enable_METIS_NodeNDP
  --sp_enable_MUMPS_SYMQAMD --sp_enable_agg_amalg`, `NDP=<nd_param>` env. `LD_LIBRARY_PATH=.../magma-2.8.0/build/lib`.
- 케이스: `exp/cases/{case3120sp,case_ACTIVSg25k,case_SyntheticUSA}`. (case_ACTIVSg70k는 현 케이스셋에 없음.)
</content>
</invoke>
