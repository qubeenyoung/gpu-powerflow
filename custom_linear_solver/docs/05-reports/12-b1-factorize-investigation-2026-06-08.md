# B=1 factorize 조사: 프로파일 + small/mid 커널 최적화 시도 (음성 결과 위주)

**작성일**: 2026-06-08
**선행**: [`docs/04/14`](../04-benchmarks-profiling/14-fp32-singlestream-baseline-2026-06-08.md), [`docs/05-reports/11`](11-tier-split-dispatch-occupancy-gate-2026-06-08.md)
**환경**: RTX 3090(sm_86, 82 SM), CUDA 12.8, fp32, single-stream. GPU 시간은 `nsys --cuda-graph-trace=node`, 커널 메트릭은 `ncu`.
**결론**: B=1의 병목은 **깊은 narrow level이 1블록·0.3% GPU util·순차 pivot chain**에 묶인 **구조적 latency**다. small/mid intra-kernel 트윅(블록 크기, tier)으로는 B=1이 안 줄어든다(측정으로 확인). 적용 변경 **없음**.

## 1. 프로파일 — B=1은 idle가 아니라 비효율 (8387)

graph 내부를 펼치니 GPU busy(~310µs) ≈ wall(~340µs). factor 분해:

| kernel | us/iter | 비중 |
|--------|--------:|----:|
| factor_mid | 239 | 77% |
| factor_small | 66 | 21% |
| scatter_values | 4.5 | 1.5% |

`factor_mid` 22개 launch가 front 수와 무관하게 **거의 다 ~8–15µs**.

## 2. ncu — single-block latency가 본질

`factor_mid`(fp32, 항상 256 threads)를 ncu로:

| level 유형 | grid | SM throughput | barrier stall |
|-----------|------|--------------:|--------------:|
| deep narrow (front 1개) | (1,1)×256 | **0.3%** | 1.5 |
| shallow (front 709/400) | (709/400,1)×256 | 9–18% | 4.9–5.8 |

- **deep narrow level = 1블록이 빈 GPU(81 SM idle)에서 실행 → 0.3% util.** etree 깊이의 직렬성(level 간 의존) + 한 level에 front 1개라 병렬화 불가. ~16개 narrow level × ~10µs ≈ factor의 절반 이상.
- shallow level은 `lu_small_front`의 pivot당 `__syncthreads`에 barrier-bound(고 throughput이 아님).

## 3. 시도 A — spine 사슬 fusion (cnt=1 chain → 1 launch)

순차 spine을 한 커널로 묶어 per-level launch 제거.
- 결과: 8387 B=1 **−3%(wall)/−7.5%(GPU커널)**, 저-B 스윗스팟 B=4 −10.7%. fused 46µs가 9 launch(~98µs) 대체.
- 한계: spine ~98µs의 대부분이 launch가 아니라 **single-block compute**라 fusion으로 안 줄어듦(gap·re-staging만 제거). **25k는 root 근처 spine front가 shared(96KB) 초과로 fusion 미발동.**
- 판정: 사용자 지시로 **제거**(spine 방향 배제).

## 4. 시도 B — small/mid 블록 크기 occupancy-aware 조정

`factor_mid`(fp32)는 항상 256 threads. front 크기에 맞춰 64/128/256으로 스케일:
- **ungated(무조건 scale down)**: high B 개선(8387 B=256 **−5%**, 25k −3%; blocks/SM occupancy↑) but **B=1 +21% 악화**. 빈 GPU의 1블록은 스레드가 많을수록 intra-block 병렬이 이득 — barrier 절감보다 serialization 패널티가 큼.
- **gated(`level_size×B > num_SMs`일 때만 scale)**: B=1 regression은 막지만 결과가 noise 수준으로 일관성 없음(8387 B=16 +6.5%, 25k B=256 +5.7%). 깔끔히 맞추려면 threshold 튜닝이 필요 — **특정 케이스 미세 최적화(숫자놀이) 금지 원칙에 위배**.
- 판정: **되돌림**. (블록 스케일은 *high-B* 최적화로는 유효하나 B=1 목표엔 부적합.)

## 5. 결론 / 방향

B=1 factorize의 지배 비용은 **깊은 narrow level의 single-block 직렬 latency**(0.3% util)다. 스레드를 늘려도 순차 pivot chain은 안 빨라지고, 줄이면 B=1이 악화 — **intra-kernel small/mid 트윅으로는 B=1이 개선되지 않는다**(측정 확인).

실효 레버는 둘 중 하나로 좁혀짐:
- **구조적 fusion**(narrow level을 device-sync로 묶기 / megakernel) — 직렬성 자체를 한 launch로. 복잡·리스크.
- **batching**(여러 Newton/contingency를 묶어 B↑) — doc 14가 보인 B≈64 포화. GPU를 채우면 narrow level도 B개 front로 병렬화됨. B=1 단일 시스템에서는 근본적으로 GPU가 빈다.

high-B를 노린다면 §4의 블록 스케일(occupancy-aware)이 −3~5% 후보로 남아있음(별도 작업).
