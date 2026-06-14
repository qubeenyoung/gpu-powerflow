# mid+big tiled-trailing 프로토타입 — 2-커널 형은 회귀 (B=1)

> **상태**: negative result (prototype iteration 1)   **갱신**: 2026-06-12
> **한 줄**: [`260612_goal.md`](../260612_goal.md) 목표 3 을 "big tiling" → "under-fill 레벨의 mid+big front 를 tiled-trailing 으로" 일반화한 2-커널 프로토타입은 실패했고, **결정적 증거는 under-fill 을 제거한 B=64 에서도 세 케이스 전부 0.62–0.68× 회귀(70K 포함)** 라는 점 — 즉 **타일링은 일감 부족이 아니라 *본질적으로* 비효율**이다(ncu: 중복 L/U staging 으로 DRAM **+28%**, launch **+57%**; thin-K front 라 staging 이 지배적 비용인데 타일링이 이를 배수로 늘림). B=1 회귀(mid 0.66–0.72×)·중립(70K 1.007×)은 이 고유 비효율에 under-fill·shared-residency 손실이 겹친 결과. occupancy 천장은 실재하나 **tiling 은 그걸 회수하는 잘못된 도구**임을 정량 확정한다.

## 동기 (목표 3 재정의)

[`../20260612_lab_meeting/factorize-bottleneck-3case.md`](../20260612_lab_meeting/factorize-bottleneck-3case.md): mid·big 전 레벨이 1 block/SM(warp 25–33%), batch-invariant. **13K·25K 는 big 이 없는 mid-dominated** 라, big 만 다룬 [`../deprecated/big_split_2d/`](../../deprecated/big_split_2d/)(usa B=1 1.14×, 70K 0.95× 회귀)는 이들을 건드리지 못했다. → under-fill 레벨의 front 를 mid/big 무관하게 tiled-trailing 으로 보내는 일반화를 테스트.

## 구현

- [`src/factorize/tiled.cuh`](../../src/factorize/tiled.cuh) (env-gated, default OFF): `factor_tiled_panel`(Phase1+2, 1 block/front, global F) → `factor_tiled_trail_tf32`(Phase3, grid `(fronts,B,64×64 타일)`, 타일당 1블록, ~16 KB shared, fused extend-add). v2 `trailing_tile_tf32` 포팅.
- gate ([`src/factorize/schedule.cuh`](../../src/factorize/schedule.cuh) `issue_factor_level_range` kLarge): `CLS_TILED_TRAILING=1` + TF32 + `level_size×B < CLS_TILED_FILL`(default #SM). mid·big 공통 — front 는 항상 global 버퍼에 있고 mid 는 *staging*만 shared 로 하므로 동일 구조가 둘 다 커버.
- 빌드: `CLS_INTERNAL_GRAPH=OFF`(graph 우회, launch 1:1). B=1, `--no-multistream`, repeat 30 / warmup 10.

## 측정 — factorize ms (B=1, tf32)

| case | regime | baseline | tiled (best fill) | **speedup** |
|---|---|---:|---:|---:|
| 13659 | mid-dominated (big 없음) | 0.399 | 0.554 | **0.72× 회귀** |
| ACTIVSg25k | mid-dominated (big 없음) | 0.582 | 0.877 | **0.66× 회귀** |
| ACTIVSg70k | big-dominated | 1.712 | 1.701 | 1.007× 중립 |

fill ∈ {82, 200, 512} 전부 회귀/중립(best 표기). → **mid-dominated 는 명확히 회귀, big-dominated 는 중립.**

## ncu — 왜 안 되나 (occupancy *한계* ≠ *달성*)

70K tiled trailing 커널(under-fill 레벨, grid `(fronts,1,4–9)`):

| | achieved warp% | 의미 |
|---|---:|---|
| baseline `factor_big` | 26–27 | 1 block/SM (기존 3case 문서) |
| **`factor_tiled_trail_tf32`** | **6–14 (median 7.6)** | **오히려 더 낮음** |

big_split_2d 가 보고한 "1→9-10 block/SM" 는 `launch__occupancy_limit_blocks`(이론 *한계*)다. 그러나 **달성 occupancy(`warps_active`)는 6–14%** — under-fill 레벨은 front 가 2–10개뿐이라 타일로 쪼개도 (grid 236블록·4 warp/블록) **커널이 너무 짧고 일감이 적어** steady-state 에 못 가고 launch/tail latency 가 지배. 즉 **occupancy 한계를 풀어도 달성 occupancy 가 안 오른다 — 일감이 없어서.**

## 분석 — 두 가지 손실

1. **mid 일반화의 shared-residency 손실 (신규)**: production mid 커널은 front 전체를 빠른 shared 에서 인수분해(`factor_mid_blocked`). tiled 경로는 panel 을 **global** 에서 돌린다. mid front 는 uc 가 작아(13K/25K max uc≈80 → 2×2=4 타일) 타일 fan-out 이득이 미미한데 shared 이점을 통째로 버려 → **순손실**. 13K/25K 가 순수 mid 라 전면 회귀.
2. **2nd-launch / 짧은 커널 (big_split_2d 재확인 + 심화)**: panel→trailing 2-커널의 per-level 2nd launch + global L/U 왕복. graph-OFF 라 최대로 노출되지만, **핵심은 graph 와 무관** — tiled trailing 커널 자체의 achieved warp 6–14% 는 일감 부족이라 graph 로도 안 풀린다.

## 추가 검증 — 타일링 *고유* 비효율 (under-fill 변수 분리)

위 B=1 결론은 "under-fill(일감 부족)"과 "타일링 자체의 비효율"을 섞을 위험이 있었다. 둘을 분리하려고
**GPU 가 꽉 차는 B=64(under-fill 없음)에서 타일링을 강제 ON**(`CLS_TILED_FILL=1e8`)해 baseline 과 비교:

| case | regime | base B=64 (ms/sys) | tiled 강제 B=64 | **speedup** |
|---|---|---:|---:|---:|
| 13659 | mid-dom | 0.0439 | 0.0652 | **0.67× 회귀** |
| ACTIVSg25k | mid-dom | 0.0950 | 0.1542 | **0.62× 회귀** |
| ACTIVSg70k | **big-dom** | 0.3714 | 0.5498 | **0.68× 회귀** |

→ **under-fill 이 없는데도(B=64) 세 케이스 전부 0.62–0.68× 회귀** — B=1 에서 중립이던 **70K(big-dom)까지**.
즉 회귀는 일감 부족 때문이 아니라 **타일링 자체가 본질적으로 비효율**이다.

**메커니즘 (ncu, 25K B=64, factor 커널 전체 합)**:

| | launches | DRAM 트래픽 | kernel time |
|---|---:|---:|---:|
| baseline | 68 | 2029 MB | 5406 µs |
| tiled 강제 | 107 (**+57%**) | **2600 MB (+28%)** | 9302 µs (**1.72× 느림**) |

타일링은 **더 많은 일을 한다**: 출력 타일마다 자기 L-행·U-열을 global 에서 shared 로 **다시 staging**
(monolithic 은 front 를 한 번만 staging) → **DRAM +28%**, 거기에 panel-in-global 왕복 + per-level 2nd
launch(**+57% launches**). B≥16 은 메모리 대역폭 bound 라 이 +28% 트래픽이 곧장 느려짐.

**왜 thin-K 에서 특히 나쁜가**: 타일 1개의 trailing 계산 = `uc_tile²·nc`, staging = `uc_tile·nc + nc·uc_tile`.
계산/staging ≈ `uc_tile`. 전력망 front 는 **nc 작고(thin-K) uc_tile 도 작아** staging 이 지배 →
타일링이 *지배적 비용(staging)을 배수로 늘린다*. monolithic 은 한 번 staging 후 재사용해 이 비용을 피한다.

## 결론 — tiling 은 이 워크로드에 *본질적으로* 비효율 (winning regime 없음)

| regime | 왜 tiling 이 지나 |
|---|---|
| **B=64 full** | **타일링 고유 비효율** — 중복 staging DRAM +28%, launch +57% → **0.62–0.68×, 전 케이스(70K 포함)**. 본 측정. |
| **B=1 under-fill** | 위 고유 비효율 + 일감 부족(달성 occupancy 6–14%). big-dom 70K 만 under-fill 이득이 고유 비효율을 상쇄해 *중립*(이득 아님). |
| **mid-sized front** | global panel → shared-residency 손실까지 추가. mid-dom 전면 회귀. |

→ **big_split_2d(big-only, usa 1.14×)의 그 1.14× 도 under-fill 이득이 고유 비효율을 *겨우* 넘은 narrow
한 점**이었고(70K 0.95×), 본 측정은 그 고유 비효율을 **+28% DRAM·+57% launch 로 정량화**한다. occupancy
천장(1 block/SM)은 실재하나 **tiling 은 그걸 회수하는 잘못된 도구** — front 를 블록으로 재분배하는 비용
(shared 재-staging)이 thin-K 에선 *지배적 비용*이라, 어떤 occupancy 이득보다 크다.

B=1 factorize 의 latency 뿌리(깊고 좁은 etree 상위 front 2–10개)는 여전히 **합치는(launch/barrier 감소)**
방향으로 풀어야 하지만, **tiling 은 그 도구가 아님이 이제 정량적으로 확정**됐다.

## 다음 단계 — 재설정

1. **cooperative single-kernel(`grid.sync`)** 는 2nd launch 만 없앨 뿐, **달성 occupancy 6–14%(일감 부족)는 못 고친다** → 본 측정상 B=1 under-fill 에선 상한이 낮아 우선순위 강등.
2. **방향 전환 후보** (목표 3 재정의):
   - **B=1 launch/barrier 감소**: 깊은 etree 상위(front 2–10) 의 인접 레벨/spine 을 **단일 커널로 fuse**(레벨당 launch 제거). 이게 under-fill latency 의 진짜 뿌리.
   - **batch 로 under-fill 흡수**: B=1 은 본질적으로 GPU 를 못 채움 → 처리량은 B≥16 + 메모리(staging) 트래픽 절감이 레버(bottleneck §B≥16). 단일 시스템 latency 와 분리.
   - mid occupancy(목표 2): mid 가 grid 충분에도 33% 인 건 whole-front shared + thread 수(128–512) → tiling 아닌 **thread/shared 점유 튜닝**.

## 코드 처리

`src/factorize/tiled.cuh` 는 env-gated OFF 라 production 무영향. cooperative 를 추진하지 않으면 big_split_2d 처럼 `deprecated/` 로 이동 권장(본 노트가 측정 근거). 게이트 통합부(schedule.cuh kLarge)도 함께.

## 재현
```bash
cmake -S custom_linear_solver -B build -DCLS_INTERNAL_GRAPH=OFF -DCLS_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
CASE=/path/to/case_ACTIVSg25k
build/custom_linear_solver_run $CASE --precision tf32 --repeat 30 --warmup 10 --no-multistream                          # baseline
CLS_TILED_TRAILING=1 CLS_TILED_FILL=82 build/custom_linear_solver_run $CASE --precision tf32 --repeat 30 --warmup 10 --no-multistream  # tiled
```
