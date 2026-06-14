# mid __syncthreads 감소(warp-sync / blocked-LU) 프로토타입 — barrier 는 증상, occupancy 가 천장

> **상태**: negative result (prototype iteration 2)   **갱신**: 2026-06-12
> **한 줄**: mid 가 B=1 에서 barrier stall 5.5 로 압도적이라 `__syncthreads` 를 줄이는 두 방향((3) warp-owned 대각블록, (2) sync-free U-solve)을 시도했으나, (3)은 **load imbalance 로 barrier 를 5.2→11.2 로 악화**(0.90×), (2)는 **barrier 를 5.36→5.12 로 거의 못 줄이고 wall 무변**(neutral) — panel-LU 의 per-pivot sync 가 barrier 의 대부분인데 직렬화 없이는 못 없애고, **warp%가 31.8→31.9 로 불변**이라 결국 **barrier 는 1 block/SM occupancy 의 *증상*이지 *레버*가 아님**이 확인됐다. tiling 에 이어 mid 도 kernel-level 탈출구가 없음을 보강한다.

## 동기 (ceiling GREEN 이었음)

[mid stall 측정] `factor_mid_blocked`, 25k B=1: **barrier 5.5 압도적**(wait 2.46, long_sb 1.97). mid 는 메모리·compute 아닌 `__syncthreads` 대기 bound. tiling 과 달리 **monolithic 유지 → staging 안 늘림**(07 노트의 +28% DRAM 함정 회피). 그래서 sync 감소가 순이득일 것으로 예상.

baseline TF32 mid 경로(`factorize_front_blocked_tf32`)는 BK=8 블록마다 panel kk-loop(per-pivot `__syncthreads`) + U-solve kk-loop(per-row `__syncthreads`) → ~2·nc block barriers.

## 시도 1 — (3) warp-owned 대각블록 (blocked-LU)

warp 0 가 kb×kb 대각블록만 `__syncwarp` 로 factor, tall L-panel(B)·U-panel(C)은 thread 별 독립 패스(sync 없음). BK 블록당 `__syncthreads` ~3개(이론상 2·nc→3).

**결과 (B=1, tf32): 13659 0.99× · 25k 0.90× · 70k 0.96× — 회귀.**

ncu (25k): **barrier 5.17 → 11.24 (악화)**, dur 539→643us. → warp 0 가 대각블록 도는 동안 **나머지 7 warp 가 `__syncthreads`(1)에서 대기** → load imbalance. **1 block/SM 은 barrier *개수*가 아니라 *imbalance*를 처벌한다.** warp-serialize 는 정확히 그 imbalance 를 키워 역효과.

## 시도 2 — (2) sync-free U-solve only (직렬화 없이)

(3)의 imbalance 를 빼고, **balanced panel(baseline 그대로) + U-solve 만 sync-free**(열 j 별 thread 가 kb-row forward-subst 를 자기 체인으로 → 열 독립이라 sync 불필요). nc 개 U-solve barrier 를 직렬화 없이 제거.

**결과 (B=1, tf32): 13659 0.99× · 25k 1.02× · 70k 0.94× — neutral~경미한 회귀.**

ncu (25k): **barrier 5.36 → 5.12 (거의 안 줄음)**, dur 541→520us(노이즈), **warp% 31.8→31.9(불변)**.

→ **U-solve sync 는 barrier stall 의 0.24 뿐** — barrier 의 대부분은 **panel-LU 의 per-pivot sync**다. 그건 직렬화 없이 못 없애고(시도 1 이 그래서 역효과), 없앨 수 있는 U-solve 만 빼도 **wall 이 안 움직인다**.

## 분석 — barrier 는 occupancy 의 증상

| 실험 | barrier stall | warp%(occ) | wall |
|---|---:|---:|---:|
| baseline | 5.36 | 31.8 | 1.00× |
| (3) warp-owned | **11.24** | 31.7 | 0.90× |
| (2) sync-free U-solve | 5.12 | 31.9 | 1.01× |

**warp% 가 셋 다 ~32% 로 못 박혀 있다** — mid 는 deep level 에서 512 thread(16 warp) × **1 block/SM**(whole-front shared `fsz²·4` 가 99 KB 예산을 거의 다 씀, fsz≥112 면 1 block). 16/48 = 33% 가 천장. 이 1 block/SM 에선 **어떤 warp 가 barrier(또는 dependency, memory)에서 멈춰도 SM 을 채울 다른 warp 가 없다** → barrier stall 이 높게 *측정*되지만, 그건 occupancy 부족의 **증상**이다. barrier 를 없애면 warp 는 그냥 **다음 stall(wait/panel barrier)로 옮겨 갈 뿐**(시도 2: barrier 빠진 만큼 dur 거의 그대로, occ 불변).

즉 **ceiling-first 가 "barrier 5.5 압도적"이라 GREEN 으로 보였지만, barrier-dominant ≠ barrier-가-레버**. barrier 는 1 block/SM 의 결과이고, 진짜 binding 은 **whole-front shared staging 이 강제하는 1 block/SM occupancy** 다.

## 결론 — mid 도 kernel-level 탈출구 없음 (tiling 과 같은 벽)

| 방향 | 결과 | 왜 |
|---|---|---|
| tiling (07) | 회귀 | front 쪼개면 staging 중복(+28% DRAM), occupancy 한계≠달성 |
| (3) warp-sync 대각 | 회귀 | 7 warp idle-wait → barrier imbalance 악화 |
| (2) sync-free U-solve | neutral | barrier 줄여도 occupancy(32%) 불변 → wall 무변 |

세 시도 모두 **binding constraint = 1 block/SM occupancy(whole-front shared)** 를 못 건드린다. occupancy 를 올리려면 whole-front 를 shared 에 안 올려야 하는데, 그 shared-residency 가 바로 mid 를 빠르게 만드는 것(07 에서 mid front 를 global panel 로 뺐다가 회귀 확인). **자가당착**: mid 의 속도원천(shared)이 곧 occupancy 천장.

→ B=1 mid/big 은 구조적으로 occupancy-bound 이고 **factorize-kernel micro-opt(tiling·sync 감소)로는 못 푼다.** 진짜 레버는 여전히 kernel 밖 — **batch(B≥16 처리량)** 또는 **ordering(critical-path 단축, reorder-lab)**. (cls 결론: [`../260612_goal.md`](../260612_goal.md) §3 재설정 표 참조.)

## 부산물 — 그래도 건진 것
- **sync-free U-solve(C)는 correctness OK·neutral~경미 개선**: 직렬화 없이 nc U-solve barrier 제거. 손해 아니므로, 후일 occupancy 가 풀리는 맥락(예: 작은 mid front 가 2 block/SM 되는 경우)에서는 미세 이득 가능. 현재는 무영향이라 default OFF 유지.
- **방법론 교훈**: stall-dominant 지표는 *레버가 아니라 증상*일 수 있다 — occupancy(warp%)가 그 micro-opt 로 안 움직이면 wall 도 안 움직인다. 다음부터 ceiling-first 에 **"이 레버가 warp% 를 올리는가?"** 를 같이 본다.

## 코드 처리
`factorize_front_blocked_tf32_fewsync`(+`CLS_MID_FEWSYNC` 게이트) 는 env-gated OFF, production 무영향. sync-free U-solve 만 남아 correctness-안전·neutral. 추진 안 하면 tiled.cuh 와 함께 `deprecated/` 이동 후보.

## 재현
```bash
CASE=/path/to/case_ACTIVSg25k
build/custom_linear_solver_run $CASE --precision tf32 --repeat 40 --warmup 15 --no-multistream            # baseline
CLS_MID_FEWSYNC=1 build/custom_linear_solver_run $CASE --precision tf32 --repeat 40 --warmup 15 --no-multistream  # sync-free U-solve
```
