# B=1 factorize: profiling + spine-chain fusion (phase 1)

**작성일**: 2026-06-08
**선행**: [`docs/04/14`](../04-benchmarks-profiling/14-fp32-singlestream-baseline-2026-06-08.md), [`docs/05-reports/11`](11-tier-split-dispatch-occupancy-gate-2026-06-08.md)
**대상**: `src/factorize/kernels.cuh`(`factor_spine_chain`), `src/factorize/dispatch.cuh`
**환경**: RTX 3090(sm_86, 82 SM), CUDA 12.8, fp32, single-stream. A/B는 interleaved+median, GPU 시간은 `nsys --cuda-graph-trace=node`(graph 내부 커널 노출).

## 0. TL;DR

B=1은 GPU가 **idle가 아니라 실제로 바쁜데(busy≈wall) 비효율**이다. 8387 B=1 factor ~310µs 중 `factor_mid` 77%, 그리고 **mid level launch는 front 수와 무관하게 거의 다 ~8–15µs** — 깊은 narrow level이 **빈 GPU에서 1블록 직렬 실행되는 single-block latency**에 묶임.

phase-1로 **cnt=1 spine 사슬을 한 커널로 fuse**(저-B 게이트). 결과:
- **8387 B=1 factor −3%(wall) / −7.5%(GPU 커널시간)**; 저-B 스윗스팟 **B=4 −10.7%**. fused spine 46µs가 9개 per-level launch(~98µs)를 대체.
- **25k: 효과 없음(무회귀).** spine(7 panels, root 근처)의 front가 커서 fused 커널 shared(96KB) 초과 → per-level fallback.

정확성 불변(8387 ~2–4e-5, 25k ~1.7e-4, fp64 3e-14).

## 1. 프로파일 (8387 B=1, graph-node trace)

per-iter(=total/iters) GPU busy:

| kernel | us/iter | 비중 |
|--------|--------:|----:|
| factor_mid (22 launch) | 239 | 77% |
| factor_small (6) | 66 | 21% |
| scatter_values | 4.5 | 1.5% |
| **factor 합** | **~310** | (wall ~340µs) |

핵심: 22개 `factor_mid` launch가 **front 708개 level이나 1개 level이나 ~8–15µs로 비슷**. 즉 깊은 narrow level은 compute가 아니라 **1블록이 SM 1개에서 직렬 수행되는 latency**가 지배(81 SM idle). etree가 root로 갈수록 cnt=1 사슬(spine)이 되는 일반 성질.

8387 spine = L19~27(cnt=1) 9개. `d_spine_panels`는 analyze에서 계산만 되고 **미사용**이었음.

## 2. 설계 — fused spine 커널 (저-B)

`factor_spine_chain<T>`: grid=(1, B), **한 블록이 spine 사슬을 bottom→top 순차 처리**(`panel_parent[spine[s]]==spine[s+1]`이라 직렬, 패널 간 `__syncthreads`). 각 패널마다 stage_in→panel LU+trailing(scalar staged)→writeback→부모로 extend_add. 사슬이 한 블록 안에 있어 front를 shared/L2에 warm 유지하고 per-level launch·re-staging을 제거.

**게이트(저-B 한정)**: spine level은 cnt=1이라 per-level launch가 B blocks. `B ≤ num_SMs`일 때만 fuse(그 이상이면 per-level이 이미 GPU를 채움). 추가로 사슬 최대 front의 shared가 96KB를 넘으면 fuse 안 함(→ 25k fallback). spine 커널은 below-spine level들이 spine bottom front를 다 assemble한 **뒤**에 launch.

case 상수 없음(num_SMs·shared cap은 HW값) → 일반화.

## 3. 결과 (factor ms/sys, interleaved median of 5)

| case | B=1 | B=4 | B=16 | B=64 | B=256 |
|------|----:|----:|-----:|-----:|------:|
| case8387 | −3.1% | **−10.7%** | −3.7% | −1.5% | +0.8%* |
| ACTIVSg25k | −0.2%† | −3.2%† | −0.6%† | −3.9%† | −4.2%*† |

`*` B=256은 게이트로 fuse 안 함(B>SM). `†` 25k는 spine fusion 미발동(shared 초과) → 변화는 noise.

GPU 커널시간(8387 B=1): factor numeric 305→282µs(−7.5%); fused spine 46µs가 9 launch(~98µs)를 대체. wall(−3%)과 차이는 클럭 미고정 noise.

## 4. 정직한 한계 — 왜 작은가

spine ~98µs의 **대부분은 launch 오버헤드가 아니라 single-block compute**다(9개 front를 1블록이 순차 factor). fusion은 inter-launch gap + re-staging만 제거(~−50µs on spine, 총 factor의 ~−7%). **B=1의 진짜 벽인 narrow level의 single-block compute 자체는 줄지 않는다.**

- 25k처럼 root 근처 front가 큰 대형 그리드는 spine이 shared에 안 들어가 fusion 자체가 안 됨.

## 5. 후속 (phase 2 후보, 본 변경 범위 밖)
- **narrow non-spine level fusion**: L12~18(cnt 2~12)을 device-side 동기화로 묶기 — cross-front 의존 때문에 더 복잡.
- **single-block compute 가속**: 깊은 mid front를 1블록이 더 빨리(더 많은 thread/overlap) — 상한 제한적(1 SM).
- **대형 그리드용 spine**: shared 안 들어가는 큰 spine front를 global-resident로 fuse하는 별도 커널.
- 큰 그림: B=1은 etree 깊이의 직렬성이 본질이라, 여러 Newton iteration/contingency를 **배치로 묶어 B를 올리는 것**이 가장 큰 레버(doc 14가 보인 B=64 포화).
